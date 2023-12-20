import asyncio
import websockets
import os
import shutil
from Llama2 import ChatGenerator

talkmate = ChatGenerator()

async def new_client_connected(client_socket, path):
    client_address = str(client_socket.remote_address[1])
    print("New client connected", client_address)
    directory_name = 'generated_files/' + client_address+'/'
    os.makedirs(directory_name, exist_ok=True)

    try:
        while True:
            msg = await client_socket.recv()
            talkmate.generate(msg, client_socket, directory_name)
    except websockets.exceptions.ConnectionClosedOK:
        shutil.rmtree(directory_name)
        print(f"Client {client_address} disconnected")

async def start_server():
    print("Server started")
    server = await websockets.serve(new_client_connected, 'localhost', 12345)

    try:
        await server.wait_closed()
    except KeyboardInterrupt:
        server.close()
        await server.wait_closed()

if __name__ == '__main__':
    event_loop = asyncio.get_event_loop()

    try:
        event_loop.run_until_complete(start_server())
        event_loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        event_loop.close()
