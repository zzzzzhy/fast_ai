import sys
import numpy
import cffi
ffi = cffi.FFI()

with open('src/convert.h') as my_header:
    ffi.cdef(my_header.read())

with open('src/convert.c') as my_source:
    if __debug__:
        print('Building the debug build...')
        ffi.set_source(
            '_convert',
            my_source.read(),
            extra_compile_args=['-pedantic', '-Wall', '-g', '-O0','-I/root/darknet/include'],
            extra_link_args=['-ldarknet']
        )
    else:
        print('Building for performance without OpenMP...')
        ffi.set_source(
            '_convert',
            my_source.read(),
            extra_compile_args=['-Ofast']
        )

ffi.compile()  # convert and compile - mandatory!

