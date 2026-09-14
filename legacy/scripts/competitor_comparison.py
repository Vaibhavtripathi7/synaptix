from google import genai
from google.genai import types
import base64

from pyxnat.core.uriutil import file_path


def generate():
  client = genai.Client(
      vertexai=True,
      project="warm-actor-451113-j8",
      location="us-central1",
  )

  text1 = types.Part.from_text(text="""Analyze two documents and provide a comprehensive comparison outlining their differences, similarities, and potential learning points for each.

**Instructions:**

1. Carefully examine `""")
  file_path1 = "ed2a395c-5e9b-4411-8b4a-a718d192155a.pdf"
  with open(file_path1, "rb") as f:  # Open local file in binary read mode
    pdf_bytes = f.read()  # Read the file content as bytes

  document1 = types.Part.from_bytes(
    data=pdf_bytes,  # Use from_bytes with the file content
    mime_type="application/pdf",
  )
  file_path2 = "download.pdf"
  with open(file_path2, "rb") as f:  # Open local file in binary read mode
    pdf_bytes = f.read()  # Read the file content as bytes

  document2 = types.Part.from_bytes(
    data=pdf_bytes,  # Use from_bytes with the file content
    mime_type="application/pdf",
  )
  text2 = types.Part.from_text(text="""`.
2. Identify the key differences between the two documents. Consider content, structure, style, and any other relevant aspects.
3. Identify the key similarities between the two documents. Look for common themes, approaches, or information.
4. Based on the identified differences and similarities, suggest how the first company can learn from the second company.  Be specific and provide actionable insights.
5. Similarly, suggest how the second company can learn from the first company.  Provide specific and actionable insights.

**Output Format:**

```
**Comparison of Document 1 and Document 2**

**Differences:**

* [List the key differences between the documents]

**Similarities:**

* [List the key similarities between the documents]

**Learning Points for Company 1 (from Company 2):**

* [List specific and actionable learning points]

**Learning Points for Company 2 (from Company 1):**

* [List specific and actionable learning points]
```

**Document 1:**



**Document 2:**""")

  model = "gemini-2.0-flash-exp"
  contents = [
    types.Content(
      role="user",
      parts=[
        text1,
        document1,
        types.Part.from_text(text="""` and `"""),
        document2,
        text2
      ]
    )
  ]
  generate_content_config = types.GenerateContentConfig(
    temperature = 1,
    top_p = 0.95,
    max_output_tokens = 8192,
    response_modalities = ["TEXT"],
    safety_settings = [types.SafetySetting(
      category="HARM_CATEGORY_HATE_SPEECH",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_DANGEROUS_CONTENT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_HARASSMENT",
      threshold="OFF"
    )],
  )

  for chunk in client.models.generate_content_stream(
    model = model,
    contents = contents,
    config = generate_content_config,
    ):
    print(chunk.text, end="")

generate()