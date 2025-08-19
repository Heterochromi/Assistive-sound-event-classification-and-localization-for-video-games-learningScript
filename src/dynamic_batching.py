import torch
import gc

def find_optimal_batch_size(model, input_shape, initial_batch_size, device, memory_usage_fraction=0.95):
    """
    Finds the optimal batch size that fits into a given fraction of the available GPU memory.

    Args:
        model (torch.nn.Module): The model to test.
        input_shape (tuple): The shape of the input tensor for a single sample (e.g., (C, H, W)).
        initial_batch_size (int): The batch size to start searching from.
        device (torch.device): The device to run the test on (e.g., 'cuda').
        memory_usage_fraction (float): The fraction of free GPU memory to target.

    Returns:
        int: The optimal batch size.
    """
    if device == 'cpu':
        print("Device is CPU, returning initial batch size.")
        return initial_batch_size

    torch.cuda.empty_cache()
    gc.collect()

    model.to(device)
    model.train()

    free_memory, _ = torch.cuda.mem_get_info(device)
    target_memory = free_memory * memory_usage_fraction
    
    print(f"Available GPU memory: {free_memory / 1024**3:.2f} GB")
    print(f"Target memory usage: {target_memory / 1024**3:.2f} GB")

    # --- Binary search for the optimal batch size ---
    low = 1
    high = initial_batch_size * 4  # Start with a reasonable upper bound
    optimal_batch_size = 1

    while low <= high:
        batch_size = (low + high) // 2
        if batch_size == 0:
            break
        
        try:
            # Create a dummy input tensor
            dummy_input = torch.randn((batch_size, *input_shape), device=device)
            
            # Forward and backward pass to measure memory usage
            output = model(dummy_input)
            loss = output.sum() # A dummy loss
            loss.backward()
            
            # Clean up
            del dummy_input, output, loss
            torch.cuda.empty_cache()
            gc.collect()
            
            # If successful, it means this batch size fits. Try a larger one.
            optimal_batch_size = batch_size
            low = batch_size + 1
            print(f"Batch size {batch_size} fits. Trying larger.")

        except RuntimeError as e:
            if 'out of memory' in str(e):
                # If it's an OOM error, this batch size is too large. Try a smaller one.
                high = batch_size - 1
                print(f"Batch size {batch_size} is too large (OOM). Trying smaller.")
                torch.cuda.empty_cache()
                gc.collect()
            else:
                # Some other runtime error
                raise e
    
    print(f"Found optimal batch size: {optimal_batch_size}")
    return optimal_batch_size
