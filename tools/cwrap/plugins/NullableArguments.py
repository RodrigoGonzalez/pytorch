from . import CWrapPlugin


class NullableArguments(CWrapPlugin):

    def process_single_check(self, code, arg, arg_accessor):
        if 'nullable' in arg and arg['nullable']:
            return f'({code} || {arg_accessor} == Py_None)'
        return code

    def process_single_unpack(self, code, arg, arg_accessor):
        if 'nullable' in arg and arg['nullable']:
            return f'({arg_accessor} == Py_None ? NULL : {code})'
        return code
