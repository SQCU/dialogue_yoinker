# Bridge Generator Prompt Template

You are generating **bridge dialogue** to connect two segments of a synthetic dialogue graph.

## Your Task

Given:
1. **Terminus segment** - the END of one dialogue chain (what comes before your bridge)
2. **Entry segment** - the START of another dialogue chain (what your bridge should lead into)
3. **Lore bible** - the setting rules and vocabulary

Generate a **single line of dialogue** (10-30 words) that:
- Flows naturally FROM the terminus context
- Leads naturally INTO the entry context
- Maintains the setting's tone and vocabulary
- Has an appropriate emotion for the transition

## Constraints

- **Brevity**: One line only. This is a transition, not a scene.
- **Neutrality**: Bridge dialogue is often transitional - acknowledgments, redirects, topic shifts.
- **Setting-appropriate**: Use Gallia vocabulary (Hexagon, Prefecture, dossier, département, etc.)
- **No new proper nouns**: Use existing setting elements, don't invent new ones.

## Output Format

Respond with ONLY a JSON object:

```json
{
  "bridge_text": "Your single line of bridge dialogue here.",
  "bridge_emotion": "neutral",
  "reasoning": "Brief explanation of why this bridges the two segments."
}
```

## Example

**Terminus** (happy): "Take the Prefecture's token! Listen to Dubois!"
**Entry** (happy): "Lunch? No, I don't believe I'm... available."

```json
{
  "bridge_text": "But before we proceed - there's the matter of the midday recess.",
  "bridge_emotion": "neutral",
  "reasoning": "Shifts from urgent instruction to pause for lunch, connecting bureaucratic action to meal-culture."
}
```

---

## Request Data

Below is the link request to process:

```json
{request_json}
```

Generate the bridge dialogue now.
