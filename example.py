# Libs
import discord
from asyncio import sleep

import transcribe_pack


# Variables
TOKEN = 'YOUR_TOKEN'


# Define clinet
intents = discord.Intents.all()
intents.members = True
intents.message_content = True
client = discord.Client(intents=intents)
channel = client.get_channel('YOUR CHANNEL ID')


@client.event
async def on_ready():
    
    # Load Whisper Model
    model = transcribe_pack.WhisperRecognizer(2) # 2 = small model
    
    last_message = None
    last_len = 0
    
    while True:
        # For loop
        await sleep(0.25)
        
        # Get transcriptions
        result = model.get_sentence()
        if result==None: continue
        
        # If new line is added:
        if not len(result) == last_len:
            # Update
            last_len = len(result)
            try: 
                # Send new message
                last_message = await channel.send(result[-1])
            except Exception as e: print(e)
            
        else: # if same length before
            try: 
                # Edit previous message
                await last_message.edit(content=result[-1])
            except Exception as e: print(e)
        


    
client.run(TOKEN)