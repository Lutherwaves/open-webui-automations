# ConSens Filter for Open WebUI

This filter implements the ConSens (Context Sensitivity) scoring system that evaluates how much context contributes to generating an answer. The ConSens filter automatically analyzes model responses and adds contextual relevance scores to conversations.

## What is ConSens?

ConSens (Context Sensitivity) is a metric that measures how much a given context influences the generation of an answer. It works by:

1. Calculating the perplexity of an answer given both question and context
2. Calculating the perplexity of the same answer given only the question (no context)  
3. Computing the ratio to determine context utilization

The score ranges from -1 to 1:
- **Positive scores** (closer to 1): Context was helpful and well-utilized
- **Negative scores** (closer to -1): Context was not helpful or potentially misleading
- **Scores near 0**: Context had neutral impact

## How It Works

Unlike a pipeline that acts as a separate model, the ConSens **Filter** automatically intercepts every conversation:

1. 🗣️ User asks a question
2. 🤖 Your chosen model generates a response  
3. 📊 ConSens filter automatically triggers
4. 🧮 Calculates context sensitivity score using the same conversation
5. ✨ Adds the ConSens analysis to the response
6. 👀 User sees the response with ConSens insights

**No manual activation needed** - it works transparently with any model!

## Installation

1. Install the required dependencies:
```bash
pip install torch transformers numpy
```

2. Place the `ConSens.py` file in your Open WebUI **functions** directory (not pipelines)

3. Restart your Open WebUI server

4. Enable the filter in **Admin Panel > Settings > Functions**

## Configuration

The filter supports the following configuration options (Valves):

- **MODEL_NAME**: HuggingFace model for perplexity calculation (default: "distilgpt2")
- **MAX_LENGTH**: Maximum sequence length for the model (default: 512)
- **DEVICE**: Device to run the model on - "auto", "cpu", or "cuda" (default: "auto")
- **SHOW_CONSENS**: Enable/disable ConSens scoring (default: true)
- **SHOW_DETAILS**: Show detailed analysis vs compact score (default: true)
- **MIN_RESPONSE_LENGTH**: Minimum response length to analyze (default: 10 characters)

## Usage Examples

### Automatic Operation

The filter works automatically with any conversation:

**User**: "What is the capital of France?"

**Assistant** (with system context about European geography):
> Based on the information provided, the capital of France is Paris. Paris is not only the capital but also the largest city in France.
> 
> ---
> **ConSens Analysis:**
> - **Score:** 0.742 (Context was helpful)
> - **Question:** What is the capital of France?
> - **Context available:** Yes
> - **Perplexity with context:** 12.45
> - **Perplexity without context:** 28.73
> 
> *ConSens measures how much context influenced this response. Positive scores indicate helpful context, negative scores suggest the context wasn't useful.*

### Configuration Examples

**Compact View** (SHOW_DETAILS = false):
> The capital of France is Paris.
> 
> **ConSens:** 0.742 (helpful)

**Disabled** (SHOW_CONSENS = false):
> The capital of France is Paris.
> *(No ConSens analysis shown)*

### Testing

Run the example script to test the filter:

```bash
python ConSens_example.py
```

## Model Selection

The default model is **DistilGPT-2**, which is lightweight and doesn't require any tokens. The filter includes automatic fallback to ensure reliability:

### Recommended Models (No Token Required)
- **distilgpt2** (default): Fastest, good for testing
- **gpt2**: Standard baseline, good balance
- **microsoft/DialoGPT-small**: Optimized for conversations

### Larger Models (if you have resources)
- **gpt2-medium**: Better accuracy, more memory
- **gpt2-large**: Highest accuracy, significant resources needed

### Important Notes
- ✅ **All recommended models are freely available** (no HuggingFace tokens needed)
- ✅ **Automatic fallback**: If your chosen model fails, it tries distilgpt2 → gpt2 → DialoGPT-small
- ❌ **Don't use embedding models** like sentence-transformers (they can't calculate perplexity)

Update the `MODEL_NAME` valve in the Open WebUI admin panel to change models.

## Performance Considerations

- **Model Size**: Larger models provide more accurate perplexity but require more memory and computation
- **Context Length**: Longer contexts may need larger `MAX_LENGTH` settings
- **Device**: GPU acceleration significantly improves performance for larger models
- **Response Length**: Very short responses (< MIN_RESPONSE_LENGTH) are skipped for efficiency

## Integration with Other Functions

The ConSens filter works alongside other Open WebUI functions:

- **Filter Priority**: Use the `priority` valve to control execution order
- **Compatible**: Works with any model (local, OpenAI, Anthropic, etc.)
- **Non-intrusive**: Only adds analysis, doesn't modify the core response
- **Efficient**: Uses lazy loading and skips analysis for very short responses

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `MAX_LENGTH` or use a smaller model
2. **Model Download Fails**: Check internet connection and HuggingFace model availability
3. **CUDA Errors**: Set `DEVICE` to "cpu" if GPU issues occur
4. **Filter Not Working**: Ensure it's placed in `functions/` directory, not `pipelines/`

### Error Messages

- `"ConSens model not available"`: Model failed to load, check dependencies
- `"ConSens calculation failed"`: Numerical issues in calculation  
- `"ConSens error"`: General filter error, check logs

### Performance Tips

- Use `distilgpt2` for fastest results
- Set `SHOW_DETAILS=false` for cleaner output
- Increase `MIN_RESPONSE_LENGTH` to skip analysis of short responses
- Use `DEVICE=cpu` if you encounter GPU memory issues

## Technical Details

The filter uses the following approach:

1. **Message Parsing**: Extracts the latest user question and assistant response
2. **Context Extraction**: Gathers conversation history and system messages as context
3. **Perplexity Calculation**: Uses the negative log-likelihood of the sequence
4. **Context Comparison**: Compares perplexity with and without context
5. **Score Transformation**: Applies sigmoid transformation for -1 to 1 range
6. **Response Modification**: Adds ConSens analysis to the assistant's message

The formula for ConSens is:
```
r = log(pe / pc)
consens = 2 / (1 + exp(-r)) - 1
```

Where:
- `pe` = perplexity without context
- `pc` = perplexity with context

## Comparison: Filter vs Pipeline

| Aspect | ConSens Filter | Pipeline |
|--------|---------------|----------|
| **Activation** | Automatic with every response | Manual model selection |
| **Integration** | Transparent, works with any model | Separate model in UI |
| **Usage** | Zero configuration needed | Requires switching models |
| **Performance** | Runs after response generation | Replaces response generation |
| **Flexibility** | Works with all providers | Limited to pipeline setup |

## License

This filter is provided as-is for research and educational purposes. 