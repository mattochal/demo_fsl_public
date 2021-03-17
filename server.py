import argparse
import time
import socket
import pickle
import struct
import pprint
import copy
import pdb
import tqdm
import json

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.join(os.path.dirname(currentdir),'src')
sys.path.insert(0,parentdir)

from utils.utils import set_torch_seed, set_gpu, get_tasks, get_data, get_model, get_backbone, get_strategy, get_main_parser
from utils.utils import compress_and_print, torch_summarize, extract_args_from_file, Bunch, get_raw_args, toDict
from utils.builder import ExperimentBuilder 
from utils.ptracker import PerformanceTracker
from utils.dataloader import DataLoader
from tasks.task_generator import TaskGenerator
from fsl_demo_task import DemoFSLTask

import sys
import pprint
import numpy as np


def setup_algorithms(server_args):
    """
    Load datasets and pretrained models
    """
    
    loaded_models = {}
    datasets = None
    abspath = os.path.abspath(".")
    set_torch_seed(server_args.seed)
    
    if "exp_path" in server_args and server_args["exp_path"] is not None:
        abspath = os.path.abspath(server_args["exp_path"])
        
    for builder_args in server_args.models:
        original_args = copy.copy(builder_args)
        
        assert 'continue_from' in builder_args, 'All "models" should have a "continue_from" entry.'
        assert 'gpu' in builder_args, 'All "models" should have a specified "gpu" entry or "cpu" device.'
        
        stdin_list = [
            "--continue_from", os.path.join(abspath, builder_args["continue_from"]),
            "--gpu", builder_args['gpu'],
            "--seed", server_args.seed,
            "--dataset", server_args.dataset,
            "--dataset_args", json.dumps({'test': {'dataset_version': server_args.version,
                                                   'data_path': server_args.data_path}})
        ]
        
        builder_parser = get_main_parser()
        builder_args = get_raw_args(builder_parser, stdin_list=stdin_list, args_dict={})
        
        device   = set_gpu(builder_args.gpu)
        tasks    = get_tasks(builder_args)
        datasets = get_data(builder_args) if datasets is None else datasets
        backbone = get_backbone(builder_args, device)
        strategy = get_strategy(builder_args, device)
        model    = get_model(backbone, tasks, datasets, strategy, builder_args, device)
        compress_and_print(builder_args)
        
        system = ExperimentBuilder(model, tasks, datasets, device, builder_args)
        system.load_pretrained()
        
        model.set_mode('test')
        
        if builder_args["model"] == 'simpleshot':
            system.model.set_train_mean(system.datasets['train'])
        
        name = original_args['name'] if 'name' in original_args else builder_args['model']
        tie_breaker = 0
        name_proposal = name
        while name_proposal in loaded_models:
            tie_breaker+=1
            name_proposal = "{}({})".format(name,tie_breaker)
        loaded_models[name_proposal] = system
        
    return loaded_models, datasets
    
def setup_socket(server_args):
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the port
    server_address = ('localhost', server_args.port)
    print('starting up on {} port {}'.format(*server_address))
    sock.bind(server_address)

    # Listen for incoming connections
    sock.listen(1)
    return sock

def send_data(sock, data):
    print('sending', data)
    pickled = pickle.dumps(data)
    send_one_message(sock, pickled)

def send_one_message(sock, data):
    length = len(data)
    sock.sendall(struct.pack('!I', length))
    time.sleep(0.5)
    sock.sendall(data)

def recv_one_message(sock):
    lengthbuf = recvall(sock, 4)
    if not lengthbuf: return None
    length, = struct.unpack('!I', lengthbuf)
    data = recvall(sock, length)
    return data

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def wait_for_response(sock):
    print('waiting for a response')
    data = recv_one_message(sock)
    if data is None:
        return None
    return pickle.loads(data)


class Server:
    
    def __init__(self, server_args):
        self.server_args = server_args
        self.sock = setup_socket(self.server_args)
        self.systems, self.datasets = setup_algorithms(self.server_args)
        self.ptracker_args = {
            "test": { "metrics": [ "accuracy", "loss", "preds" ], "save_task_performance": False },
            "train": { "metrics": [ "accuracy", "loss", "preds" ], "save_task_performance": False },
            "val": { "metrics": [ "accuracy", "loss", "preds" ], "save_task_performance": False }
        }
        
    def listen(self):
        while True:
            # wait for client:
            print("\n\n\n")
            print("--------------------------------------------")
            print('| Waiting for a connection on port: {} |'.format(self.server_args.port))
            print("--------------------------------------------")
            connection, client_address = self.sock.accept()
            print('> Connection accepted!')
            self.restore_initial_state()

            try: 
                self.listen_and_reply(connection)

            finally:
                # Clean up the connection
                connection.close()
        
    def listen_and_reply(self, sock):
        while True:
            print('listening')
            message = wait_for_response(sock)
            
            if message is None:
                break

            if type(message) is dict and 'action' in message:

                if message['action'] == 'setup':
                    data = self.setup_information()
                    send_data(sock, data)

                if message['action'] == 'classify':
                    data = self.train_algorithms(message['supports'], message['targets'])
                    send_data(sock, data)

                if message['action'] == 'reset_task':
                    print('reset task')
                    self.restore_initial_state()

            else:
                print('Message unrecognised! Message: {}'.format(message))

    def setup_information(self):
        data = {
            'action':'setup',
            'algorithms': list(self.systems.keys()),
            'dataset': self.server_args.dataset,
            'version': self.server_args.version,
            'dataset_sig': self.datasets['test'].get_signature(),
            'dataset_args': {setname:toDict(self.datasets[setname].args) for setname in ['train', 'test', 'val']}
        }
        return data

    def train_algorithms(self, supports_idx, targets_idx):
        data = {'action':'output','models':{}}
        
        with tqdm.tqdm(total=len(self.systems), disable=False) as pbar_val:
            for model_name in self.systems:
                ptracker = PerformanceTracker(args=self.ptracker_args)
                builder = self.systems[model_name]
                
                supports_lblname = [builder.datasets['test'].inv_class_dict[i] for i in supports_idx]
                targets_lblname  = [builder.datasets['test'].inv_class_dict[i] for i in targets_idx]
                
                slbl_uniq, supports_lbl =  np.unique(supports_lblname, return_inverse=True)
                
                tlbl_uniq = np.array(slbl_uniq.tolist())
                tlbl_uniq_map = {n:i for i, n in enumerate(tlbl_uniq)}
                targets_lbl = np.array([tlbl_uniq_map[name] for name in targets_lblname])
                
                task_args = {"test":{"support_idx":supports_idx,"support_lbls":supports_lbl,
                                     "target_idx":targets_idx,  "target_lbls" :targets_lbl}}
                
                print("training {} on {} supports, eval {} targets".format(model_name, len(supports_idx), 
                                                                           len(targets_idx)))
                ptracker.set_mode('test')
                builder.model.set_mode('test')
                builder.task_args['test'] = Bunch(task_args['test'])
                task_generator = TaskGenerator(builder.datasets['test'],
                                               task=DemoFSLTask,
                                               task_args=Bunch(task_args['test']),
                                               num_tasks=1,
                                               seed=builder.args.seed, 
                                               epoch=builder.state.epoch,
                                               mode='test', 
                                               fix_classes=False,
                                               deterministic=True)
                for sampler in task_generator:
                    dataloader = DataLoader(builder.datasets['test'], sampler, builder.device, builder.state.epoch, 'test') 
                    builder.model.meta_test(dataloader, ptracker)
                    
                pbar_val.set_description('Testing ({}) -> {} {}'. format(model_name,
                                            ptracker.get_performance_str(),
                                            builder.model.get_summary_str()))
                data['models'][model_name] = tlbl_uniq[ptracker.lastest_task_performance["preds"]].tolist()
        return data

    def restore_initial_state(self):
        for model_name in self.systems:
            self.systems[model_name].model.net_reset()



def get_server_parser(server_parser=argparse.ArgumentParser()):
    server_parser = argparse.ArgumentParser(description="Incremental FSL Demo",prog='server')
    server_parser.add_argument('--args_file', type=str, default='server_args.json', 
                               help="Path and filename to json configuration file, over writing the values in the argparse")
    server_parser.add_argument('--models', type=list, default=[])
    server_parser.add_argument('--port', type=int, default=8991)
    server_parser.add_argument('--dataset', type=str, default='ssss')
    server_parser.add_argument('--version', type=str, default=None)
    server_parser.add_argument('--seed', type=int, default=0)
    server_parser.add_argument('--data_path', default='../data/')
    server_parser.add_argument('--exp_path',  default='../experiment/')
    return server_parser


def get_server_args(args, to_ignore_from_file=["model", "task"], update_args={}, arg_list=None):
    args_dict = vars(args)
    args_dict.update(update_args)
    
    if args.args_file not in ["None", None, ""]: 
        # The args passed with the command take priority over file args
        to_ignore = [arg[2:] for arg in sys.argv if arg.startswith("--")]
        to_ignore.extend(to_ignore_from_file)
        print("To ignore from file", to_ignore)
        args_dict.update(extract_args_from_file(args.args_file, to_ignore=to_ignore))
    
    pprint.pprint(args_dict, indent=4)
    args = Bunch(args_dict)
    return args


def main(server_args):
    # setup
    print('Setting up server')
    time.sleep(2)
    server = Server(server_args)
    server.listen()
    
    
if __name__ == "__main__":
    server_parser = get_server_parser()
    server_args = get_server_args(server_parser.parse_args())
    main(server_args)