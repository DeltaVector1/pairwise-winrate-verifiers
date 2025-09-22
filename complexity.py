import json
import argparse
import spacy
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count

MAX_PARTICIPLES = 2
MAX_NESTED_CLAUSES = 2
ANALYZE_FIELD = "text"
nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])

def analyze_complexity(text):
    doc = nlp(text)
    analysis = {
        "has_passive_voice": False,
        "max_nested_clauses": 0,
        "longest_sentence": 0,
        "num_participle_phrases": 0,
        "num_sentences": len(list(doc.sents)),
        "is_complex": False,
        "reasons": [],
        "score": 0,
    }
    for sent in doc.sents:
        for token in sent:
            if token.dep_ in ["nsubjpass", "auxpass"]:
                if not analysis["has_passive_voice"]:
                    analysis["reasons"].append("Contains passive voice")
                    analysis["has_passive_voice"] = True
                    analysis["score"] += 1
                break
        sent_participles = sum(1 for token in sent if token.tag_ in ["VBG", "VBN"] and token.dep_ in ["ROOT", "advcl", "acl"])
        sent_clauses = sum(1 for token in sent if token.dep_ in ["ccomp", "xcomp", "advcl", "relcl"])
        analysis["max_nested_clauses"] = max(analysis["max_nested_clauses"], sent_clauses)
        analysis["longest_sentence"] = max(analysis["longest_sentence"], len(sent))
        analysis["num_participle_phrases"] = max(analysis["num_participle_phrases"], sent_participles)
        if sent_participles > MAX_PARTICIPLES:
            analysis["reasons"].append(f"Too many participle phrases ({sent_participles}) in one sentence")
            analysis["score"] += (sent_participles - MAX_PARTICIPLES)
        if sent_clauses > MAX_NESTED_CLAUSES:
            analysis["reasons"].append(f"Deeply nested clauses ({sent_clauses}) in one sentence")
            analysis["score"] += (sent_clauses - MAX_NESTED_CLAUSES)
    analysis["is_complex"] = analysis["score"] > 0
    return analysis

def process_line(args):
    """ Process a single JSONL entry """
    line, filter_out_complex, threshold = args
    obj = json.loads(line)
    if ANALYZE_FIELD in obj and obj[ANALYZE_FIELD]:
        complexity_analysis = analyze_complexity(obj[ANALYZE_FIELD])
        complexity_score = complexity_analysis["score"]
    else:
        complexity_analysis = {
            "is_complex": False,
            "reasons": [f"MISSING FIELD {ANALYZE_FIELD}"],
            "has_passive_voice": False,
            "max_nested_clauses": 0,
            "longest_sentence": 0,
            "num_participle_phrases": 0,
            "score": 0
        }
        complexity_score = 0
    # Original object without modification
    result = json.dumps(obj, ensure_ascii=False)
    # For threshold report
    if not filter_out_complex or complexity_score <= threshold:
        return result, complexity_score
    return None, complexity_score  # Skip saving this log

def process_jsonl(input_file, output_file, filter_out_complex, num_workers, threshold=None):
    input_path = Path(input_file)
    complexity_scores = []
    with input_path.open('r') as f_in, open(output_file, 'w') as f_out:
        total_lines = sum(1 for _ in f_in)
        f_in.seek(0)
        with Pool(num_workers) as pool:
            with tqdm(total=total_lines, desc="Processing logs") as pbar:
                for result, score in pool.imap_unordered(process_line, ((line, filter_out_complex, threshold) for line in f_in), chunksize=10):
                    complexity_scores.append(score)
                    if result:
                        f_out.write(result + '\n')
                    pbar.update(1)
    return complexity_scores

def main():
    parser = argparse.ArgumentParser(description=f"Analyze and filter {ANALYZE_FIELD} field in JSONL file.")
    parser.add_argument('input_file', type=str, help='Input JSONL file path')
    parser.add_argument('-f', '--filter', action='store_true', help="Filter out logs exceeding complexity threshold")
    parser.add_argument('-w', '--workers', type=int, default=cpu_count(), help="Number of worker processes")
    parser.add_argument('-t', '--threshold', type=int, help="Filter by complexity threshold (skips report generation)")
    parser.add_argument('-r', '--range', type=str, default="0,32,1", 
                      help="Threshold range for report as start,end,step (default: 0,32,1)")
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    
    if not input_path.exists():
        raise RuntimeError(f"File '{input_path}' does not exist")
    
    # Parse threshold range
    try:
        range_parts = list(map(int, args.range.split(',')))
        if len(range_parts) == 3:
            start, end, step = range_parts
        elif len(range_parts) == 2:
            start, end = range_parts
            step = 1
        else:
            start = range_parts[0]
            end = start + 32
            step = 1
    except ValueError:
        print(f"Invalid range format: {args.range}. Using default 0,32,1")
        start, end, step = 0, 32, 1
    
    if args.threshold is not None:
        output_path = input_path.with_stem(f"{input_path.stem}_threshold_{args.threshold}")
        print(f"Processing '{input_path}' with {args.workers} workers (Output: '{output_path}')")
        print(f"Filtering entries with complexity score <= {args.threshold}")
        process_jsonl(input_path, output_path, args.filter, args.workers, args.threshold)
    else:
        # Generate threshold report
        output_path = input_path.with_stem(f"{input_path.stem}_analyzed")
        print(f"Processing '{input_path}' with {args.workers} workers (Output: '{output_path}')")
        complexity_scores = process_jsonl(input_path, output_path, False, args.workers)
        
        print("\nGenerating threshold report...")
        print("\nThreshold\tKept Items\tPercentage")
        print("----------------------------------------")
        total_items = len(complexity_scores)
        
        for threshold in range(start, end, step):
            kept = sum(1 for score in complexity_scores if score <= threshold)
            percentage = (kept / total_items) * 100
            print(f"{threshold}\t\t{kept}\t\t{percentage:.1f}%")
    
    print("DONE")

if __name__ == "__main__":
    main()