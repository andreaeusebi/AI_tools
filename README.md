# AI_tools

Repo hosting tools and utilities scripts to for AI related tasks.

## Requirements
TBD.

## Installation

Run following command (set the proper path to the _AI_tools_ folder):

``` 
python3 -m pip install -e /path/to/AI_tools
``` 

## Usage

In a Python script simply import and then use the module you need as follows:

```
import AI_tools.huggingface 
from AI_tools.segments_ai.segments2hfdataset import segments2HfDataset

...

AI_tools.huggingface.hf_segformer_demo.displayInferenceResults(...)
segments2HfDataset(...)
``` 


## VSCode setup to avoid Import errors

1. Open Command Palette with _Ctrl+Shift+P_.
2. Select _Preferences: Open Workspace Settings_.
3. Scroll until _Python_ section.
4. Look for _Auto Complete: Extra Paths_ and click on _Edit in settings.json_.
5. In the field _python.autoComplete.extraPaths_ add the path to the AI_tools/ folder.