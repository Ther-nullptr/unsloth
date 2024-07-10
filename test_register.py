import torch

if __name__ == '__main__':
    # step 1: test the matmul
    a = torch.randn(10, 512, 4096 * 4).cuda().to(torch.bfloat16)
    b = torch.randn(4096 * 4, 4096).cuda().to(torch.bfloat16)
    
    a_small = torch.randn(10000, 16).cuda().to(torch.bfloat16)
    b_small = torch.randn(16, 10000).cuda().to(torch.bfloat16)
    
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    
    # calculate the time    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    c1 = torch.matmul(a, b)

    start.record()
    for i in range(1000):
        c = torch.matmul(a, b)
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()
    
    print(f'total time: {start.elapsed_time(end)}')
    
    a = a * (a > 2.)
    
    # get the sparse ratio
    print(f'sparse ratio: {torch.sum(a > 2.5).item() / a.numel()}')
    
    start.record()
    for i in range(1000):
        a_sparse = a.to_sparse()
    end.record()
    
    torch.cuda.synchronize()

    print(f'total time: {start.elapsed_time(end)}')