import os
import sys
from subprocess import Popen, call, PIPE, STDOUT

CWD = os.path.realpath(os.getcwd())

# MPI = "mpirun"
# MPI = "/opt/openmpi-gcc/bin/mpirun"

HOST_FILE_NAME = "compute_host"

USE_HOSTFILE = not os.path.isfile("/.dockerenv")


def find_MPI():
    p = Popen("which mpirun", stdout=PIPE, stdin=PIPE, stderr=PIPE, shell=True)
    p_out, p_err = p.communicate(input=None)
    if p.returncode != 0:
        print p_err
        raise Exception("find MPI failed")
    return p_out.strip()


def run_cmd(cmd, input):
    p = Popen(cmd, stdout=PIPE, stdin=PIPE, stderr=PIPE, shell=False)
    p_out, p_err = p.communicate(input=input)
    return p_out, p_err, p.returncode


# compute_list = [2, 3, 4, 5]
compute_list = [1, 2, 4, 5]


def gen_host_file(n_slot_each, host_file_name):
    """
    compute2 slots=8
    compute3 slots=8
    compute4 slots=8
    compute5 slots=8
    """
    
    # TODO hard-code number of machine
    compute_name_list = ["compute" + str(c) for c in compute_list]
    host_file = "\n".join([c + " slots=" + str(n_slot_each) for c in compute_name_list])

    with open(host_file_name, "w") as f:
        f.write(host_file)
        f.write("\n\n")


def make():
    cmd = ["make"]
    print " ".join(cmd)
    return run_cmd(cmd=cmd, input="")


def get_output_file_name(scale, degree, n_proc_each, executable):
    base = "output"
    s = "s" + str(scale)  # scale
    d = "d" + str(degree)  # degree
    c = "c" + "".join([str(c) for c in compute_list])  # compute list
    p = "p" + str(n_proc_each)  # number of process each node
    if USE_HOSTFILE:
        return ".".join([base, executable, s, d, c, p])
    else:
        return ".".join([base, executable, s, d, p])


def show_stdx(p_out, show_out, p_err, show_err, returncode, msg):
    if p_out is not None and show_out:
        print p_out
    if p_err is not None and show_err:
        sys.stderr.write(p_err + "\n")
    if returncode != 0:
        raise Exception(msg)


def run(scale, degree, n_proc_each, executable, do_compile):
    """

    :param scale:
    :param degree:
    :param n_proc_each:
    :param executable: program binary to run, compiled by make
    :return:
    """

    pow_of_two = ((n_proc_each & (n_proc_each - 1)) == 0) and n_proc_each > 0
    if not pow_of_two:
        raise Exception("n_proc_each is not of power of two")

    if do_compile:
        p_out, p_err, returncode = make()
        if returncode != 0:
            sys.stderr.write(p_err + "\n")
        show_stdx(p_out=p_out, show_out=False, p_err=p_err, show_err=False, returncode=returncode, msg="make failed")

    full_helper_message = ["-mca", "orte_base_help_aggregate", "0"]
    run_root = "--allow-run-as-root" if True else ""
    host_file = "--hostfile"
    np = "-np"
    output_file = get_output_file_name(scale=scale, degree=degree, n_proc_each=n_proc_each, executable=executable)
    redirect = "&>"
    n_proc_total = n_proc_each
    if USE_HOSTFILE:
        n_proc_total = len(compute_list) * n_proc_each
        gen_host_file(n_slot_each=n_proc_each, host_file_name=CWD + "/" + HOST_FILE_NAME)

    # /opt/openmpi-gcc/bin/mpirun --allow-run-as-root --hostfile host4 -np 32 graph500_mpi_simple 28 16 &> output.s28.d16.c2345.p8
    MPI = find_MPI()
    executable = CWD + "/" + executable
    if USE_HOSTFILE:
        cmd = [MPI, run_root, host_file, HOST_FILE_NAME, np, str(n_proc_total), executable, str(scale), str(degree), redirect, output_file]
    else:
        cmd = [MPI, run_root, np, str(n_proc_total), executable, str(scale), str(degree), redirect, output_file]
    cmd[1:1] = full_helper_message

    cmd = " ".join(cmd)
    print cmd

    p = Popen(cmd, shell=True)
    print "pid: " + str(p.pid)
        
    # TODO module load mpi/openmpi-x86_64
    
    # p_out, p_err, returncode = run_cmd(cmd, "")
    # show_stdx(p_out=p_out, show_out=True, p_err=p_err, show_err=True, returncode=returncode, msg="run failed")


def make_wrapper(args):
    p_out, p_err, returncode = make()
    show_stdx(p_out=p_out, show_out=False, p_err=p_err, show_err=True, returncode=returncode, msg="make failed")


def run_wrapper(args):
    run(args.scale, args.degree, args.numproc, args.executable, False)


def altogether(args):
    run(args.scale, args.degree, args.numproc, args.executable, True)
