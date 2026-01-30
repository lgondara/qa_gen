import argparse
from src.workflow import CurationWorkflow

def main():
    parser = argparse.ArgumentParser(description="Agentic Dataset Curation")
    parser.add_argument("--input_file", type=str, help="Path to input text file")
    parser.add_argument("--output_file", type=str, default="output.json", help="Path to output file")
    parser.add_argument("--demo_text", type=str, help="Direct text input for demo")
    
    args = parser.parse_args()
    
    workflow = CurationWorkflow()
    
    if args.demo_text:
        result = workflow.process_item(args.demo_text)
        print("\n=== FINAL RESULT ===")
        print(result)
        return

    if args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Here we might want to chunk the text if it's large, 
        # but for this simple framework, we treat the whole file as one source context 
        # or just take the first chunk.
        result = workflow.process_item(text)
        
        if result:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"Saved to {args.output_file}")
        else:
            print("No data generated (discarded).")

if __name__ == "__main__":
    main()
