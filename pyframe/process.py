# coding=utf-8
"""Run jobs in parallel"""

import os
import sys
import socket
import time
import subprocess as sp
import multiprocessing as mp
import multiprocessing.managers

__all__ = ['run', 'process_jobs', 'setup_calculations']


def run(command):
    process = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
    stdout, stderr = process.communicate()
    return stdout, stderr


def process_jobs(directories, filenames, node_list, jobs_per_node, comm_port):

    """Parallel processing of jobs as given in a shell script. The locations and names of the
    scripts are in directories and filenames."""

    try:
        auth_key = os.urandom(8)
    except NotImplementedError:
        auth_key = b'PyFraME'

    manager = create_queue_manager(comm_port, auth_key=auth_key)
    manager.start()

    job_queue = manager.get_job_queue()
    result_queue = manager.get_result_queue()

    time.sleep(3)

    start_clients(node_list, comm_port, auth_key, jobs_per_node)

    for directory, filename in zip(directories, filenames):
        assert os.path.isdir(directory)
        assert os.path.isfile(os.path.join(directory, filename))
        os.chmod(os.path.join(directory, filename), 0o744)
        command = 'bash -c -l \'cd {0}; time -p ./{1}\''.format(directory, filename)
        job_queue.put((filename, command))

    num_jobs = len(filenames)
    results = []
    while num_jobs > 0:
        filename, result = result_queue.get()
        results.append((filename, result))
        num_jobs -= 1

    time.sleep(3)

    job_queue.close()

    manager.shutdown()
    shutdown_clients()

    timings = {}
    for filename, (stdout, stderr) in results:
        output = stderr.decode().split()
        try:
            timings[filename] = output[output.index('real')+1]
        except ValueError:
            timings[filename] = 'N/A'
    with open('PyFraME_timings.out', 'a') as timings_file:
        for filename in filenames:
            timings_file.write('{0}:\n Wall time {1} seconds\n\n'.format(filename, timings[filename]))
    with open('PyFraME.stdout', 'a') as output_file:
        for filename, (stdout, stderr) in results:
            if stdout:
                output_file.write('{0}:\n{1}\n'.format(filename, stdout.decode()))
    with open('PyFraME.stderr', 'a') as error_file:
        for filename, (stdout, stderr) in results:
            if stderr:
                error_file.write('{0}:\n{1}\n'.format(filename, stderr.decode()))


def create_queue_manager(comm_port, auth_key):

    class QueueManager(multiprocessing.managers.BaseManager):
        pass

    job_queue = mp.Queue()
    result_queue = mp.Queue()
    QueueManager.register('get_job_queue', callable=lambda: job_queue)
    QueueManager.register('get_result_queue', callable=lambda: result_queue)
    manager = QueueManager(address=('', comm_port), authkey=auth_key)
    return manager


def start_clients(node_list, comm_port, auth_key, jobs_per_node):

    create_client_script(comm_port, auth_key, jobs_per_node)
    remote_command = 'bash -c -l \'cd {0}; nohup {1} PyFraME_client.py' \
                     ' >& /dev/null 2> /dev/null &\''.format(os.getcwd(), sys.executable)
    jobs = []
    for node in node_list:
        command = 'ssh {0} \"{1}\"'.format(node, remote_command)
        job = mp.Process(target=run, args=(command,))
        job.start()
        jobs.append(job)
    for job in jobs:
        job.join()


def shutdown_clients():

    os.remove('PyFraME_client.py')


def create_client_script(comm_port, auth_key, jobs_per_node):

    client_script = """import time
import subprocess as sp
import multiprocessing as mp
from multiprocessing.managers import BaseManager


def run_client():
    manager = start_client_manager('{0}', {1}, {2})
    job_queue = manager.get_job_queue()
    result_queue = manager.get_result_queue()
    slave_driver(job_queue, result_queue, {3})


def start_client_manager(master_name, comm_port, authkey):
    class QueueManager(BaseManager):
        pass
    QueueManager.register('get_job_queue')
    QueueManager.register('get_result_queue')
    manager = QueueManager(address=(master_name, comm_port), authkey=authkey)
    manager.connect()
    return manager


def slave_driver(job_queue, result_queue, jobs_per_node):
    jobs = []
    for i in range(jobs_per_node):
        job = mp.Process(target=slave, args=(job_queue, result_queue))
        job.start()
        jobs.append(job)
    for job in jobs:
        job.join()


def slave(job_queue, result_queue):
    while True:
        try:
            filename, command = job_queue.get()
            result = run(command)
            result_queue.put((filename, result))
        except Queue.Empty:
            time.sleep(3)
            result_queue.close()
            return


def run(command):
    process = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
    output, error = process.communicate()
    return output, error


if __name__ == '__main__':
    run_client()
""".format(socket.gethostname(), comm_port, auth_key, jobs_per_node)

    with open('PyFraME_client.py', 'w') as client_file:
        client_file.write(client_script)
