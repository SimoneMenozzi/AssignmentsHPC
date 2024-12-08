Run:
make EXT_CFLAGS='-DPOLYBENCH_TIME -DEXTRALARGE_DATASET' clean all run
make EXT_CFLAGS='-DPOLYBENCH_TIME -DLARGE_DATASET' clean all run
make EXT_CFLAGS='-DPOLYBENCH_TIME' clean all run
make EXT_CFLAGS='-DPOLYBENCH_TIME -DSMALL_DATASET' clean all run
make EXT_CFLAGS='-DPOLYBENCH_TIME -DMINI_DATASET' clean all run

Profile:
make EXT_CFLAGS='-DPOLYBENCH_TIME' clean all profile


the make file must overwrite the existing common.mk in the utilities folder.
Polybench.c has been modified for the name of a pointer.If it give error in the last rows of the file
probably it's because a pointer called new giving that error.
