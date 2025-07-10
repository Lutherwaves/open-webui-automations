"""
title: ConSens Filter
description: A filter that calculates ConSens (Context Sensitivity) scores for model responses based on paper https://arxiv.org/abs/2505.00065v1
author: Martin Yankov
version: 1.0.0
requirements: torch, transformers, numpy

WHAT THIS DOES:
This Filter automatically analyzes every model response in Open WebUI and calculates a 
ConSens (Context Sensitivity) score that measures how much the conversation context 
influenced the response. It works transparently with any model (OpenAI, local, etc.).

INSTALLATION:
1. pip install torch transformers numpy
2. Place this file in your Open WebUI 'functions/' directory (NOT pipelines/)
3. Restart Open WebUI
4. Enable the filter in Admin Panel > Settings > Functions

HOW IT WORKS:
- User asks a question with potential context (system message, chat history)
- Model generates response normally
- ConSens filter automatically calculates context sensitivity score
- Adds ConSens analysis to the response
- User sees response + ConSens insights

CONSENS SCORE MEANING:
- Positive (0 to 1): Context was helpful for generating the response
- Negative (-1 to 0): Context was not helpful or misleading  
- Near 0: Context had neutral impact

EXAMPLE OUTPUT:
"The capital of France is Paris.

---
**ConSens Analysis:**
- **Score:** 0.742 (Context was helpful)
- **Question:** What is the capital of France?
- **Context available:** Yes

*ConSens measures how much context influenced this response.*"
"""

from pydantic import BaseModel, Field
from typing import Optional, Callable, Any, Awaitable, Dict, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import logging
import time
import json


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=5, description="Priority level for the filter operations."
        )
        MODEL_NAME: str = Field(
            default="distilgpt2",
            description="HuggingFace model name for perplexity calculation (distilgpt2, gpt2, microsoft/DialoGPT-small)"
        )
        MAX_LENGTH: int = Field(
            default=512,
            description="Maximum sequence length for the model"
        )
        DEVICE: str = Field(
            default="auto",
            description="Device to run the model on (auto, cpu, cuda)"
        )
        SHOW_CONSENS: bool = Field(
            default=True,
            description="Show ConSens score in the response"
        )
        SHOW_DETAILS: bool = Field(
            default=True,
            description="Show detailed ConSens analysis"
        )
        MIN_RESPONSE_LENGTH: int = Field(
            default=10,
            description="Minimum response length to calculate ConSens (in characters)"
        )
        DEBUG_MODE: bool = Field(
            default=False,
            description="Enable verbose debug logging for troubleshooting"
        )
        SIMPLE_FALLBACK: bool = Field(
            default=True,
            description="Use simple heuristic if model calculation fails"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.model = None
        self.tokenizer = None
        self.device = None
        self._model_loaded = False
        
        # Enable debug logging
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("ConSens Filter initialized")
        
        # Pre-load model for better performance
        try:
            self._ensure_model_loaded()
            logging.info("ConSens: Model pre-loaded successfully")
        except Exception as e:
            logging.warning(f"ConSens: Failed to pre-load model, will load on first use: {e}")

    def _ensure_model_loaded(self):
        """Ensure the model is loaded (lazy loading)"""
        if self._model_loaded:
            return True
            
        try:
            self._load_model()
            self._model_loaded = True
            return True
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            return False

    def _load_model(self):
        """Load the language model and tokenizer with fallback options"""
        # List of reliable, lightweight models (in order of preference)
        fallback_models = [
            self.valves.MODEL_NAME,
            "distilgpt2",
            "gpt2", 
            "microsoft/DialoGPT-small"
        ]
        
        # Remove duplicates while preserving order
        models_to_try = []
        for model in fallback_models:
            if model not in models_to_try:
                models_to_try.append(model)
        
        # Determine device
        if self.valves.DEVICE == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.valves.DEVICE
        
        last_error = None
        
        for model_name in models_to_try:
            try:
                logging.info(f"Attempting to load ConSens model {model_name} on {self.device}")
                
                # Load tokenizer first (cheaper operation)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Load model
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    trust_remote_code=False  # Security: don't execute remote code
                )
                self.model.to(self.device)
                self.model.eval()

                # Add padding token if it doesn't exist
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                logging.info(f"Successfully loaded ConSens model: {model_name}")
                # Update the valve to reflect what was actually loaded
                self.valves.MODEL_NAME = model_name
                return
                
            except Exception as e:
                last_error = e
                logging.warning(f"Failed to load ConSens model {model_name}: {e}")
                # Clean up partial loads
                self.model = None
                self.tokenizer = None
                continue
        
        # If we get here, all models failed
        error_msg = f"Failed to load any ConSens model. Last error: {last_error}"
        logging.error(error_msg)
        raise RuntimeError(error_msg)

    def get_perplexity(self, question: str, context: str, answer: str) -> float:
        """
        Calculate perplexity of answer given question and context
        
        Args:
            question: The user's question
            context: The context information
            answer: The answer to evaluate
            
        Returns:
            float: Perplexity score
        """
        # Ensure model is loaded
        if not self._ensure_model_loaded():
            logging.error("ConSens model not available for perplexity calculation")
            return float('inf')
        
        # Validate inputs
        if not question.strip() or not answer.strip():
            logging.warning("Empty question or answer provided to ConSens")
            return float('inf')
        
        try:
            # Construct the prompt - simpler format for better model compatibility
            if context.strip():
                prompt = f"{context}\n\nQ: {question}\nA: {answer}"
            else:
                prompt = f"Q: {question}\nA: {answer}"

            # Debug logging
            logging.debug(f"ConSens: Computing perplexity for prompt (length: {len(prompt)})")
            
            # Tokenize the prompt with better parameters
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=min(self.valves.MAX_LENGTH, 256),  # Shorter for speed
                truncation=True,
                padding=True,
                add_special_tokens=True
            )
            
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            # Check token count
            token_count = input_ids.shape[1]
            logging.debug(f"ConSens: Tokenized to {token_count} tokens")
            
            if token_count < 3:  # Too short to be meaningful
                logging.warning("ConSens: Prompt too short after tokenization")
                return float('inf')

            # Calculate perplexity with better error handling
            with torch.no_grad():
                try:
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
                    loss = outputs.loss
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        logging.warning("ConSens: Model returned NaN or Inf loss")
                        return float('inf')
                    
                    perplexity = torch.exp(loss)
                    result = float(perplexity.cpu())
                    
                    logging.debug(f"ConSens: Raw perplexity: {result}")
                    
                    # Sanity check for the result
                    if result <= 0 or result > 10000 or torch.isnan(perplexity) or torch.isinf(perplexity):
                        logging.warning(f"ConSens: Invalid perplexity {result}, returning high value")
                        return 1000.0  # Return high but finite value
                        
                    return result
                    
                except RuntimeError as e:
                    logging.error(f"ConSens: Model runtime error: {e}")
                    return float('inf')

        except Exception as e:
            logging.error(f"Error calculating perplexity in ConSens: {e}", exc_info=True)
            return float('inf')  # Return high perplexity on error

    def calculate_consens(self, question: str, context: str, answer: str) -> Dict:
        """
        Calculate the ConSens score for the answer
        
        Args:
            question: The user's question
            context: The context information
            answer: The answer to evaluate
            
        Returns:
            dict: ConSens analysis with score and details
        """
        try:
            logging.debug("ConSens: Starting perplexity calculations...")
            
            # Calculate perplexity with context
            pc = self.get_perplexity(question, context, answer)
            logging.debug(f"ConSens: Perplexity with context: {pc}")
            
            # Calculate perplexity without context
            pe = self.get_perplexity(question, "", answer)
            logging.debug(f"ConSens: Perplexity without context: {pe}")
            
            # Check for invalid perplexities
            if pc == float('inf') and pe == float('inf'):
                logging.warning("ConSens: Both perplexities are infinite")
                return {
                    "score": 0.0,
                    "status": "both_perplexities_infinite",
                    "perplexity_with_context": pc,
                    "perplexity_without_context": pe,
                    "context_helpful": "unknown"
                }
            
            # Handle cases where one perplexity is infinite
            if pc == float('inf'):
                logging.warning("ConSens: Perplexity with context is infinite, assuming context was very unhelpful")
                return {
                    "score": -0.8,  # Very negative score
                    "status": "context_perplexity_infinite",
                    "perplexity_with_context": pc,
                    "perplexity_without_context": pe,
                    "context_helpful": "very_unhelpful"
                }
            
            if pe == float('inf'):
                logging.warning("ConSens: Perplexity without context is infinite, assuming context was very helpful")
                return {
                    "score": 0.8,  # Very positive score
                    "status": "no_context_perplexity_infinite", 
                    "perplexity_with_context": pc,
                    "perplexity_without_context": pe,
                    "context_helpful": "very_helpful"
                }
            
            # Both perplexities are finite, calculate ratio
            if pc <= 0 or pe <= 0:
                logging.warning(f"ConSens: Invalid perplexity values: pc={pc}, pe={pe}")
                return {
                    "score": 0.0,
                    "status": "invalid_perplexity_values",
                    "perplexity_with_context": pc,
                    "perplexity_without_context": pe,
                    "context_helpful": "unknown"
                }
            
            # Calculate the ratio and ConSens score
            try:
                r = np.log(pe / pc)
                logging.debug(f"ConSens: Log ratio r = log({pe}/{pc}) = {r}")
                
                if np.isnan(r) or np.isinf(r):
                    logging.warning(f"ConSens: Invalid ratio: r={r}")
                    return {
                        "score": 0.0,
                        "status": "invalid_ratio",
                        "perplexity_with_context": pc,
                        "perplexity_without_context": pe,
                        "context_helpful": "unknown"
                    }
                
                consens = 2 / (1 + np.exp(-r)) - 1
                logging.debug(f"ConSens: Final score: {consens}")
                
                if np.isnan(consens) or np.isinf(consens):
                    logging.warning(f"ConSens: Invalid final score: {consens}")
                    return {
                        "score": 0.0,
                        "status": "invalid_final_score",
                        "perplexity_with_context": pc,
                        "perplexity_without_context": pe,
                        "context_helpful": "unknown"
                    }
                
                # Determine if context was helpful
                context_helpful = "helpful" if consens > 0.1 else "neutral" if consens > -0.1 else "unhelpful"
                
                return {
                    "score": float(consens),
                    "status": "success",
                    "perplexity_with_context": pc,
                    "perplexity_without_context": pe,
                    "context_helpful": context_helpful,
                    "ratio": float(r)
                }
                
            except (ValueError, OverflowError) as e:
                logging.error(f"ConSens: Mathematical error in score calculation: {e}")
                return {
                    "score": 0.0,
                    "status": f"math_error: {str(e)}",
                    "perplexity_with_context": pc,
                    "perplexity_without_context": pe,
                    "context_helpful": "error"
                }
            
        except Exception as e:
            logging.error(f"Error calculating ConSens: {e}", exc_info=True)
            return {
                "score": 0.0,
                "status": f"error: {str(e)}",
                "perplexity_with_context": float('inf'),
                "perplexity_without_context": float('inf'),
                "context_helpful": "error"
            }

    def inlet(self, body: dict):
        """Process incoming request before sending to the model."""
        # No processing needed for inlet in ConSens filter
        return body

    def _simple_fallback_calculation(self, question: str, context: str, answer: str) -> Dict:
        """
        Simple heuristic-based ConSens calculation when model fails
        """
        try:
            # Simple heuristics based on text characteristics
            has_context = bool(context.strip())
            question_len = len(question.split())
            answer_len = len(answer.split())
            context_len = len(context.split()) if has_context else 0
            
            # Basic scoring based on text properties
            if not has_context:
                score = 0.0  # No context to evaluate
                helpful = "neutral"
            elif context_len > question_len * 2:  # Substantial context
                # Look for shared words between context and answer
                context_words = set(context.lower().split())
                answer_words = set(answer.lower().split())
                overlap = len(context_words.intersection(answer_words))
                
                if overlap > max(3, len(answer_words) * 0.1):  # Significant overlap
                    score = 0.4  # Positive but not too high
                    helpful = "helpful"
                else:
                    score = -0.2  # Some negative impact
                    helpful = "unhelpful"
            else:
                score = 0.1  # Small positive impact for minimal context
                helpful = "neutral"
            
            return {
                "score": score,
                "status": "simple_fallback",
                "perplexity_with_context": "N/A",
                "perplexity_without_context": "N/A", 
                "context_helpful": helpful,
                "method": "heuristic"
            }
            
        except Exception as e:
            logging.error(f"Simple fallback calculation failed: {e}")
            return {
                "score": 0.0,
                "status": "fallback_error",
                "perplexity_with_context": "N/A",
                "perplexity_without_context": "N/A",
                "context_helpful": "unknown"
            }

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        model: Optional[dict] = None,
    ) -> dict:
        """Process outgoing response and add ConSens analysis."""
        
        if not self.valves.SHOW_CONSENS:
            return body
        
        try:
            start_time = time.time()
            
            # Get the conversation messages
            messages = body.get("messages", [])
            if len(messages) < 2:
                logging.debug("ConSens: Not enough messages, skipping")
                return body
            
            # Find the last user message and assistant message
            user_message = None
            assistant_message = None
            context_messages = []
            
            # Parse messages in reverse to get the latest exchange
            for i in range(len(messages) - 1, -1, -1):
                msg = messages[i]
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if role == "assistant" and assistant_message is None:
                    assistant_message = content
                elif role == "user" and user_message is None:
                    user_message = content
                elif role in ["system", "user", "assistant"] and assistant_message is not None and user_message is not None:
                    # Collect earlier messages as context
                    context_messages.append(f"{role}: {content}")
            
            # Debug logging
            if self.valves.DEBUG_MODE:
                logging.debug(f"ConSens: Found user_message: {user_message[:50] if user_message else 'None'}...")
                logging.debug(f"ConSens: Found assistant_message: {assistant_message[:50] if assistant_message else 'None'}...")
                logging.debug(f"ConSens: Context messages: {len(context_messages)}")
            
            # Check if we have the required messages
            if not user_message or not assistant_message:
                logging.debug("ConSens: Missing user or assistant message, skipping")
                return body
            
            # Check minimum response length
            if len(assistant_message) < self.valves.MIN_RESPONSE_LENGTH:
                logging.debug(f"ConSens: Response too short ({len(assistant_message)} < {self.valves.MIN_RESPONSE_LENGTH}), skipping")
                return body
            
            # Prepare context from earlier messages
            context = "\n".join(reversed(context_messages))  # Reverse to get chronological order
            logging.debug(f"ConSens: Using context: {context[:100]}..." if context else "ConSens: No context available")
            
            # Send status update
            try:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Calculating ConSens score...",
                        },
                    }
                )
            except Exception as e:
                logging.warning(f"ConSens: Event emitter failed: {e}")
            
            # Calculate ConSens score
            consens_result = self.calculate_consens(user_message, context, assistant_message)
            
            # Use simple fallback if calculation failed and fallback is enabled
            if consens_result["status"] != "success" and self.valves.SIMPLE_FALLBACK:
                logging.info("ConSens: Using simple fallback calculation")
                consens_result = self._simple_fallback_calculation(user_message, context, assistant_message)
            
            calculation_time = time.time() - start_time
            logging.debug(f"ConSens calculation took {calculation_time:.2f}s")
            
            # Prepare ConSens summary
            if consens_result["status"] == "success":
                score = consens_result["score"]
                context_helpful = consens_result["context_helpful"]
                
                logging.debug(f"ConSens: Success! Score: {score:.4f}, Context: {context_helpful}")
                
                if self.valves.SHOW_DETAILS:
                    consens_text = f"""

---
**ConSens Analysis:**
- **Score:** {score:.4f} (Context was {context_helpful})
- **Question:** {user_message[:100]}{'...' if len(user_message) > 100 else ''}
- **Context available:** {'Yes' if context.strip() else 'No'}
- **Perplexity with context:** {consens_result['perplexity_with_context']:.2f}
- **Perplexity without context:** {consens_result['perplexity_without_context']:.2f}

*ConSens measures how much context influenced this response. Positive scores indicate helpful context, negative scores suggest the context wasn't useful.*"""
                else:
                    consens_text = f"\n\n**ConSens:** {score:.3f} ({context_helpful})"
                
                # Update the assistant's message
                if messages and messages[-1].get("role") == "assistant":
                    original_content = messages[-1]["content"]
                    messages[-1]["content"] = original_content + consens_text
                    logging.debug("ConSens: Successfully updated assistant message")
                
                # Send final status
                try:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"ConSens: {score:.3f} ({context_helpful}) | {calculation_time:.2f}s",
                            },
                        }
                    )
                except Exception as e:
                    logging.warning(f"ConSens: Final event emitter failed: {e}")
            else:
                # Handle calculation failure
                logging.error(f"ConSens calculation failed: {consens_result}")
                try:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"ConSens calculation failed: {consens_result['status']}",
                            },
                        }
                    )
                except Exception as e:
                    logging.warning(f"ConSens: Error event emitter failed: {e}")
        
        except Exception as e:
            logging.error(f"Error in ConSens filter outlet: {e}", exc_info=True)
            try:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"ConSens error: {str(e)}",
                        },
                    }
                )
            except Exception as ee:
                logging.warning(f"ConSens: Exception event emitter failed: {ee}")
        
        return body
