'''
Paper Reader Agent - CLI Entry Point.
'''
import sys
from rich.console import Console
from rich.markdown import Markdown
from app.agent import create_paper_agent

console = Console()

def main():
    console.print("\n[bold cyan]📚 Paper Reader Agent[/bold cyan]")
    console.print("[dim]Type your question. Commands: 'quit' to exit, 'clear' to reset.[/dim]\n")

    agent = create_paper_agent()
    messages = []

    while True:
        try:
            user_input = console.input("[bold green]You -> [/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.lower() =="clear":
            messages = []
            console.print("[dim]🗑 Conversation cleared")
            continue

        # Add user message
        messages.append({"role":"user", "content": user_input})

        # Stream agent response
        console.print("\n[bold blue] Agent -> [/bold blue]", end="")

        response_text = ""
        for step in agent.stream(
            {"messages":messages},
            stream_mode="values",
        ):
            last_msg = step["messages"][-1]
            # Printing the internal reasoning of the model
            if last_msg.type == "ai" and last_msg.tool_calls:
                # Agent decided to call a tool
                for tc in last_msg.tool_calls:
                    console.print(f"[dim yellow]🔧 Tool Call: {tc['name']}({tc['args']})[/dim yellow]")

            elif last_msg.type == "tool":
                # Tool returned a result
                preview = last_msg.content[:200] + "..." if len(last_msg.content) > 200 else last_msg.content
                console.print(f"[dim cyan]📨 Tool Result: {preview}[/dim cyan]")
            
            elif last_msg.type == "ai" and last_msg.content:
                # Final Agent response
                # Some providers (Gemini) return content as a list, not a string
                if isinstance(last_msg.content, list):
                    response_text = "\n".join(
                        block.get("text", str(block)) if isinstance(block, dict) else str(block)
                        for block in last_msg.content
                    )
                else:
                    response_text = last_msg.content
        
        # Render as markdown
        if response_text:
            console.print(Markdown(response_text))
            messages.append({"role": "assistant", "content": response_text})

        console.print()

if __name__== "__main__":
    main()
