Setup:<br/>
At least python 3.10 is required
```
pip install uv
```

```
uv run check_torch.py
```
If ```true``` continue, with following steps.

```
uv run get_window_position.py
```
- Open the game and login to any character.
- Position the cursor at the top, of the guild-only-chat-window and press "k".
- Position the cursor at the bottem, of the guild-only-chat-window and press "k".
- Note the two lines as they are important for future steps.

Run:
```
uv run main.py x1 y1 x2 y2
```

A file called "output.txt" will be generated (and updated on future runs).<br/>
Every last line from the chat, will be appended to this file.<br/>
If the chat has not changed, since the last execution, the application will output ```Images are too similar```