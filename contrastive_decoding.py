import transformers as tr
import torch

amateur_path = 'Qwen/Qwen2.5-Coder-0.5B-Instruct'
expert_path = 'Qwen/Qwen2.5-3B-Instruct'

tokenizer = tr.AutoTokenizer.from_pretrained(amateur_path)

user_message = """Give a very very brief docstring for the following function:\n```\nfunction updateEloScores(
	scores,
	results,
	kFactor = 4,
) {
	for (const result of results) {
		const { first, second, outcome } = result;
		const firstScore = scores[first] ?? 1000;
		const secondScore = scores[second] ?? 1000;

		const expectedScoreFirst = 1 / (1 + Math.pow(10, (secondScore - firstScore) / 400));
		const expectedScoreSecond = 1 / (1 + Math.pow(10, (firstScore - secondScore) / 400));
		let sa = 0.5;
		if (outcome === 1) {
			sa = 1;
		} else if (outcome === -1) {
			sa = 0;
		}
		scores[first] = firstScore + kFactor * (sa - expectedScoreFirst);
		scores[second] = secondScore + kFactor * (1 - sa - expectedScoreSecond);
	}
	return scores;
}\n```"""

prompt = tokenizer.apply_chat_template(
    [
        {'role': 'system', 'content': 'You are a helpful assistant'},
        {'role': 'user', 'content': user_message}
    ],
    add_generation_prompt=True,
    tokenize=False
)


def contrastive_generation(amateur, expert, prompt, max_tokens, alpha=0.1) -> str:
    """
    Generate text using contrastive decoding.
    
    Parameters:
      amateur: smaller  model
      expert: larger  model
      prompt: input prompt 
      max_tokens: maximum tokens to generate
      alpha: plausibility threshold hyperparameter
      
    Returns:
      generated text as a string
    """
	input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(expert.device)
	generated = input_ids.clone()
    for _ in range(max_tokens):
		with torch.no_grad():
			expert_outputs = expert(generated)
			amateur_outputs = amateur(generated)
		
		expert_logits = expert_outputs.logits[:, -1, :]
		amateur_logits = amateur_outputs.logits[:, -1, :]
		
		expert_probs = torch.softmax(expert_logits, dim=-1)
		amateur_probs = torch.softmax(amateur_logits, dim=-1)
		
		max_prob = expert_probs.max(dim=-1, keepdim=True).values
		
		allowed_mask = expert_probs >= (alpha * max_prob)
		

		contrastive_scores = torch.where(allowed_mask,
										torch.log(expert_probs + 1e-10) - torch.log(amateur_probs + 1e-10),
										torch.tensor(-float('inf')).to(expert_probs.device))
		
		next_token = contrastive_scores.argmax(dim=-1).unsqueeze(-1)
		
		generated = torch.cat([generated, next_token], dim=-1)
		
		if next_token.item() == tokenizer.eos_token_id:
			break
	# print("len",len(generated[0]))
	generated_ids_trimmed = generated[:, len(input_ids[0]):].tolist()
	return tokenizer.decode(generated_ids_trimmed[0], skip_special_tokens=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amateur_model = tr.AutoModelForCausalLM.from_pretrained(amateur_path).to(device)
expert_model = tr.AutoModelForCausalLM.from_pretrained(expert_path).to(device)
amateur_model.eval()
expert_model.eval()

output_text = contrastive_generation(amateur_model, expert_model, prompt, max_tokens=20, alpha=0.1)
print("Generated Text:\n", output_text)


