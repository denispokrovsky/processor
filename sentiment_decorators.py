import functools
from typing import Callable, Any


def sentiment_analysis_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(text: Any, *args: Any, **kwargs: Any) -> str:
        if not isinstance(text, str):
            if pd.isna(text):
                return "Neutral"  # nothing meanz neutral
            text = str(text)  # Convert to string
        
        try:
            result = func(text, *args, **kwargs)
            return result
        except Exception as e:
            print(f"Error in {func.__name__} processing text: {text[:100]}...")  # expose 100 chars of problematic text
            print(f"Error: {str(e)}")
            return "Neutral"  # nothing meanz neutral
    
    return wrapper