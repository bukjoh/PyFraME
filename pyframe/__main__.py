# coding=utf-8
"""Blablabla"""

import os
import argparse as ap

from .version import __version__


def main():

    """The main PyFraME function"""

    if 'SCRATCH' in os.environ:
        scratch_dir = os.environ['SCRATCH']
    elif os.path.isdir('/scratch'):
        scratch_dir = '/scratch'
    elif os.path.isdir('/scr'):
        scratch_dir = '/scr'
    elif os.path.isdir('/tmp'):
        scratch_dir = '/tmp'
    elif os.path.isdir('/usr/tmp'):
        scratch_dir = '/usr/tmp'
    elif os.path.isdir(os.path.join(os.getcwd(), 'scratch')):
        scratch_dir = os.path.join(os.getcwd(), 'scratch')
    elif os.path.isdir(os.path.join(os.getcwd(), 'scr')):
        scratch_dir = os.path.join(os.getcwd(), 'scr')
    elif os.path.isdir(os.path.join(os.getcwd(), 'tmp')):
        scratch_dir = os.path.join(os.getcwd(), 'tmp')
    elif os.path.isdir(os.path.join(os.getcwd(), 'temp')):
        scratch_dir = os.path.join(os.getcwd(), 'temp')
    else:
        scratch_dir = os.getcwd()

    parser = ap.ArgumentParser(prog='PyFraME', description='PyFraME {0}'.format(__version__),
                               epilog='WARNING: This is development code!',
                               usage='%(prog)s [options]', fromfile_prefix_chars='@')

    parser.add_argument('--version', action='version', version='PyFraME {0}'.format(__version__))

    parser.add_argument('-i', '--input-file', dest='input_file', metavar='FILENAME',
                        help='''Specify the name of the input file.''')

    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        help='''Print verbose output. [default: %(default)s]''')

    parser.add_argument('-w', '--work-dir', dest='work_dir', default=os.getcwd(),
                        metavar='PATH',
                        help='''Specify the path to the work directory where all permanent files
                                will be stored. [default: %(default)s]''')

    parser.add_argument('-s', '--scratch-dir', dest='scratch_dir', default=scratch_dir,
                        metavar='PATH',
                        help='''Specify the scratch directory where all temporary files will be
                                stored. It can also be set by the environment variable SCRATCH.
                                [default: %(default)s]''')

    parser.add_argument('-l', '--node-list', dest='node_list', metavar=('NODE1', 'NODE2'),
                        nargs='+', default=[],
                        help='''Provide a list of node names. Not needed if nodelist is available
                                from PBS or SLURM or if you only use the current host.''')

    parser.add_argument('-n', '--jobs-per-node', type=int, dest='jobs_per_node',  default=1,
                        metavar='NUM_JOBS_PER_NODE',
                        help='''Specify the number of jobs per node. [default: %(default)s]''')

    parser.add_argument('-m', '--memory-per-job', type=int, dest='memory_per_job',
                        metavar='MB', default=1024,
                        help='''Specify available memory per job in MB. [default: %(default)s]''')

    parser.add_argument('-t', '--omp-num-threads', type=int, default=1, dest='omp_threads_per_job',
                        metavar='OMP_NUM_THREADS',
                        help='''Sets the number of OpenMP threads. [default: %(default)s]''')

    parser.add_argument('-p', '--mpi-num-procs', type=int, default=1, dest='mpi_procs_per_job',
                        metavar='MPI_NUM_PROCS',
                        help='''Sets the number of MPI processes per job. [default: %(default)s]''')

    parser.add_argument('--port', type=int, default=5000, dest='comm_port', metavar='COMM_PORT',
                        help='''Sets the communication port number. [default: %(default)s]''')

    subparsers = parser.add_subparsers(title='subcommands')

    frag_parser = subparsers.add_parser('frag')

    frag_parser.add_argument('-s', dest='do_fragmentation', action='store_true', default=False)

    args = parser.parse_args()

if __name__ == '__main__':
    main()
