# Knowledge Assistant: user guide (C-236)

## What it does

The assistant answers questions from TODO: *the organisation's policies, procedures and knowledge base*. Every answer is built from documents you are allowed to see, and it shows the sources it used.

## How to ask

- Ask a full question: "How many days of annual leave do new staff get?" works better than "leave".
- Add details that narrow it down, such as a year, a country or a document type: "…under the 2024 travel policy".
- Ask follow-ups in the same chat. "And for part-time staff?" is understood in context.
- Click **New chat** to change topic.

## Reading an answer

| You see | It means |
|---|---|
| **[1]**, **[2]** in the text | The sentence comes from that numbered source. Open **Sources** to see the title, page and an excerpt. |
| **high confidence** | The answer is well supported by the cited sources. |
| **medium confidence** | It is partly supported. Check the sources before acting. |
| **low confidence** | The support is weak or the sources disagree. Treat the answer as a lead, not a fact. |
| "I couldn't find this…" | Nothing you can access answers the question. The topic is logged so content owners can fill the gap. |
| A clarifying question | The question was ambiguous. Answer it to continue. |
| "Needs review" | The assistant could not verify its own answer. Check with the content owner. |

## Limits

- It knows only the indexed documents. It does not know recent emails, chats or the public internet (unless web search is enabled).
- Documents are updated within TODO minutes of changing at the source.
- It can be wrong. For decisions with legal, financial or HR consequences, check the cited source or ask the owner.
- Do not paste confidential personal data into questions.

## Feedback

Use 👍 or 👎 under each answer. A 👎 with a short comment ("wrong policy year", "missing the 2024 update") is reviewed weekly, and the question is added to the test set.

## Privacy

- Your questions and answers are kept for TODO days to support follow-ups and quality reviews.
- You can delete your chat history at any time (TODO: link or button, which calls `DELETE /api/v1/conversations/me`).
