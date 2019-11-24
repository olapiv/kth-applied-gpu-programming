# Exercise 1 - CUDA Edge Detector using shared memory

Implements an edge detector for bitmap images.

## Assignment instructions for code

1. Declare a Shared Memory block within gpu_gaussian() and another one within gpu_sobel().
2. Introduce the necessary changes to make each thread bring one pixel value to the shared block. Change the input parameter of applyFilter() to use the shared block (i.e., instead of a reference to the input image directly).
3. Consider the boundaries of each thread block. Extend the Shared Memory version of gpu_gaussian() and gpu_sobel() to transfer part of the surrounding pixels of the thread block to Shared Memory. Make sure that you do not exceed the boundaries of the image.

## Q&A for report

## Acknowledgments

- Rome Photo by Willian West on Unsplash
- HK Photo by Simon Zhu on Unsplash
- NYC Photo by Matteo Catanese on Unsplash
