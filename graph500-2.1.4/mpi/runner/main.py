#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import os

import sys

import runner

def set_default_subparser(self, name, args=None):
    # http://stackoverflow.com/a/26379693/3315185
    """default subparser selection. Call after setup, just before parse_args()
    name: is the name of the subparser to call by default
    args: if set is the argument list handed to parse_args()

    , tested with 2.7, 3.2, 3.3, 3.4
    it works with 2.6 assuming argparse is installed
    """
    subparser_found = False
    for arg in sys.argv[1:]:
        if arg in ['-h', '--help']:  # global help if no subparser
            break
    else:
        for x in self._subparsers._actions:
            if not isinstance(x, argparse._SubParsersAction):
                continue
            for sp_name in x._name_parser_map.keys():
                if sp_name in sys.argv[1:]:
                    subparser_found = True
        if not subparser_found:
            # insert default in first position, this implies no
            # global options without a sub_parsers specified
            if args is None:
                sys.argv.insert(1, name)
            else:
                args.insert(0, name)


def get_args():
    argparse.ArgumentParser.set_default_subparser = set_default_subparser

    # parser = argparse.ArgumentParser(allow_abbrev=False)
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="sub-command help")

    go_parser = subparsers.add_parser("go", help="compile and run executable (default)")
    # scale, degree, n_proc_each, executable
    go_parser.add_argument("executable", type=str, help="run executable")
    go_parser.add_argument("-s", "--scale", type=int, required=True, help="specify scale (2^s vertex)")
    go_parser.add_argument("-d", "--degree", type=int, required=True, help="specify degree (vertex * d edges)")
    go_parser.add_argument("-n", "--numproc", type=int, required=True, help="specify number of process on each node")
    go_parser.set_defaults(func=runner.altogether)

    run_parser = subparsers.add_parser("run", help="run executable (default)")
    # scale, degree, n_proc_each, executable
    run_parser.add_argument("executable", type=str, help="run executable")
    run_parser.add_argument("-s", "--scale", type=int, required=True, help="specify scale (2^s vertex)")
    run_parser.add_argument("-d", "--degree", type=int, required=True, help="specify degree (vertex * d edges)")
    run_parser.add_argument("-n", "--numproc", type=int, required=True, help="specify number of process on each node")
    run_parser.set_defaults(func=runner.run_wrapper)

    rename_parser = subparsers.add_parser("compile", help="compile executable")

    # rename_parser.add_argument("sol_name", type=str, help="solution to rename")
    # rename_parser.add_argument("new_name", type=str, help="new name for solution")
    # rename_parser.add_argument("-v", "--verbose", action="store_true", help="show verbose output")
    rename_parser.set_defaults(func=runner.make_wrapper)

    parser.set_default_subparser("go")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    args.func(args)
    # runner.run(24, 16, 8, "simple")
