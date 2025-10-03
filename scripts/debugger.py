import debugpy

# Start debugpy listener
debugpy.listen(("0.0.0.0", 5678))  # or use 127.0.0.1 if debugging from same host

print("Waiting for debugger attach...")
debugpy.wait_for_client()  # Will pause here until VSCode connects
print("Debugger is attached, continuing execution.")
