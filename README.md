Visual Shopping Assistant: Full Pipeline Report
________________________________________
Approach Overview:
The Visual Shopping Assistant is a state-of-the-art multimodal AI system that assists users in interactively browsing and finding the most appropriate products from textual queries as well as visual inputs. It is based on cutting-edge AI technologies such as deep learning-based embedding models, semantic vector databases, and large language models (LLMs) to create a knowledgeable digital shop assistant. This assistant not only knows what the user is requesting but also understands images and textual metadata to generate smart and contextual product recommendations. The system architecture is based on a Retrieval-Augmented Generation (RAG) pipeline to provide real-time results..
The Visual Shopping Assistant is developed with a modular structure connecting data ingestion, document generation, semantic representation, vector search, and language generation. The design pattern adheres to the RAG paradigm to enable the system to leverage the efficiency of dense retrieval along with the expressiveness of LLMs. Below is a summary of the complete pipeline:
1.1. Data Ingestion and Merging
Various data sources are handled, such as structured Excel spreadsheets and unstructured JSON metadata. The system feeds four main CSV files:
      •description.csv :  product names and descriptions.
      •style.csv : gender, article type, base color, and more fashion characteristics.
       •image.csv: product IDs with corresponding image filenames.
       •document.csv: rich user reviews, product features, and Q&A discussions.

JSON files include rich product metadata like price, brand, usage context, and technical specifications. Each JSON file is for a product and gets parsed and joined into one all-encompassing dataframe (combine).

1.2. Unified Representation Using LangChain Documents
The data for every product both structured and unstructured is transformed into a LangChain Document. The page_content stores the concatenated text data (product description, specs, reviews), while the metadata stores structured fields like ID, brand, category, price, etc. This format supports uniform treatment for both retrieval and generation.
1.3. Embedding Layer
All documents are represented as numerical vectors that preserve semantic meanings:
•Text embeddings are calculated with transformer-based models such as all-MiniLM-L6-v2.
•CLIP is used to compute image embeddings, with the model having been trained to match images to relevant textual descriptions.
•Combined embeddings are then processed through clip-ViT-B-32 in order to match text and images to a common vector space.
1.4. Vector Indexing in Qdrant
The assistant is using Qdrant, a high-throughput vector search engine. Each product's embedded vector is indexed in a collection (multimodal_collection). This allows fast cosine similarity-based searching.
1.5. Query Handling
Whenever the user requests a question or posts a photo, the query is embedded into the same space. The system pulls the most semantically relevant products and passes them through a language model.
1.6. Response Generation based on LLM
The pulled context and query are passed through an LLM like GPT-2 or GPT-3.5-turbo, and it produces a structured, natural-sounding response. The assistant combines specifications, reviews, and highlights into a human-sounding summary.

2. Models Used 
2.1. Embedding text
• Model: all-MiniLM-L6-v2 (from sentence-transformers)
• Role: Maps sentences or paragraphs to 384-512 dimensional meaning-capturing vectors.
2.2. Image Embedding
• Model: CLIP (openai/clip-vit-base-patch32)
• Role: Embeds product images as vectors that can be matched with text queries.
2.3. Embedding Multimodal
• Model: clip-ViT-B-32 from sentence-transformers
• Role: Embeds both the image and the text in the same space to facilitate hybrid similarity searches.
2.4. Vector Database
• Tool: Qdrant
• Role: Stores and indexes all the embeddings. Performs top-k vector retrieval based on cosine similarity.
2.5. Language Models for Generation
•Base: GPT-2 (for local, quick generation)
•Advanced: GPT-3.5 Turbo (through OpenAI API) – provides improved coherence, logic, and naturalness.
These models collaborate to create a strong pipeline that can process actual e-commerce queries.

3. Instructions to Run (Detailed Guide)
Step 1: Environment Setup
•Use Google Colab for development and testing.
•Install required libraries: transformers, sentence-transformers, qdrant-client, langchain, Pillow, etc.
•Mount Google Drive to load data:
from google.colab import drive
drive.mount('/content/drive')
Step 2: Load and Merge Data
• Load CSVs with pandas.
• Parse each product's JSON file and extract fields such as price, brand, and detail.
• Merge all datasets into one DataFrame based on the product id.
Step 3: Build LangChain Documents
• Transmute each row into a Document with page_content and metadata.
• Consolidate product descriptions, reviews, and Q&A into a rich summary string.
Step 4: Embed Content
• Utilize SentenceTransformer for text.
• Utilize CLIPModel and CLIPProcessor for images.
• Optionally utilize clip-ViT-B-32 for combined embedding.
Step 5: Store in Qdrant
•Recreate collection with:
client.recreate_collection(.)
•Store each vector along with its metadata using PointStruct(id, vector, payload).
Step 6: Query Inference
•Get user text + image.
•Embed and search Qdrant.
•Get top-k relevant results.
Step 7: Generate Final Answer
•Pass obtained documents to the LLM.
•Format prompt as question + context.
•Generate structured response.

4. Sample Queries and Outputs 

Query A: "Suggest waterproof shoes for trekking."
Response:
Product: Wildcraft Trail Pro
Brand: Wildcraft
Category: Outdoor Footwear
Material: Synthetic
Waterproof: Yes
Price: ₹4,999
Recommendation: Rugged trail and wet condition friendly. Excellent grip with reinforced ankle support.


Query B: "Give me a red saree for weddings."
Response:
Product: Kanjivaram Silk Saree
Color: Red
Material: Silk with zari border
Occasion: Festive, Weddings
Price: ₹3,499
Recommendation: Traditional saree with royal appearance, ideal for cultural rituals.
Query C: Multimodal Query – Image + Text
•	Image: Screen shot of a trail running shoe
•	Text: "Is this shoe good for trail running in the rain?"
Response
Product: Adidas Trail Runner X
Category: Sports Shoes
Waterproof: Yes
Material: Mesh + rubber
Use Case: Trail running in slippery, wet ground
Price: ₹5,200
Summary: The shoe is designed for outdoor runners to provide comfort, traction, and water-resistance.

5. Real-World Use Cases
•	e-commerce Search Improvement: Replaces keyword search with semantic and image-based comprehension.
•	Product Suggestion: Users can upload product images and pose questions to discover corresponding or superior products.
• Retail Chatbots: Incorporate the assistant into a chatbot UI on shopping websites.
• Personal Shopping Assistant: Mobile apps with camera + voice input that utilize this system as backend.

6. Conclusion
Visual Shopping Assistant is a perfect demonstration of how contemporary AI can improve the customer experience. It utilizes multimodal embeddings, semantic vector search, and language model reasoning to render product discovery conversational, smart, and personalized. Being able to understand text and images, it closes the gap between product relevance and user intent — an important online shopping challenge. The convergence of Qdrant, CLIP, SentenceTransformers, and LLMs gives this a robust, end-to-end AI solution for retail use in the real world.
Challenges Encountered During Development of Visual Shopping Assistant
Development of such a powerful multimodal AI system as the Visual Shopping Assistant is not an easy task. During development, some of the most important technical challenges were encountered. These challenges mainly happened in three domains:
Data Scraping and Integration
Qdrant Vector Store Server Issues
Model and API Connectivity Issues
Let's discuss all three in detail:
1. Data Scraping and Integration Issues
Problem:
The initial product information was dispersed in several formats (JSON, CSV) with inconsistencies in field names, IDs, and missing records. JSON parsing of large quantities of files, particularly when every product existed in a distinct metadata file, was creating latency and code failures.
 Specific Problems:
Product IDs in CSVs were stored as numbers at times and strings at other times, causing merge failures.
JSON files were never there or incomplete and therefore resulted in parsing failures.
JSON fields had unexpected nested formats or lacked keys such as brand or price.
Large product descriptions along with reviews used to go beyond reasonable token limits for transformers and so text chunking was needed.
Solution:
Normalized all IDs with astype(str) prior to merging.
Applied try-except mechanisms while parsing JSON in order to catch missing or corrupted files.
Designed a fallback text template when JSON fields were absent.
Employed LangChain's RecursiveCharacterTextSplitter for chunking of long documents dynamically.
 2. Qdrant Server and Vector Storage Issues
Issue:
While running Qdrant locally (in Colab or on the desktop), server crashes and memory errors were common during large batch imports. Sometimes, Qdrant couldn't pull points, or it returned malformed data.
Particular Issues:
Mass vector inserts at once (particularly image embeddings) triggered memory spikes within the Qdrant server.
In retrieval, certain items retrieved lacked the payload attribute, causing runtime errors.
The usage of the outdated .search() method initially needed to be replaced with query_points(). 
 Solution:
Batched vector upserts into small pieces (e.g., 100 points at once).
Verified .result attribute and filtered out non-payload hits explicitly.
Used top to monitor memory usage in local deployments and employed light embedding models when necessary.
Changed over from outdated methods and consulted Qdrant's newest Python client documentation for best practices.
 3. Model Loading and API Connectivity Issues
Issue:
Big models such as GPT-2 and CLIP didn't load on free-tier Colab GPUs because of memory constraints. For a few sessions, the transformers pipeline exhibited CUDA errors. API-based models such as GPT-3.5-turbo also crashed if the API key was not set correctly.
Specific Issues:
CUDA errors such as device-side assert triggered and out of memory when trying to use big models such as gpt2-xl.
Hugging Face pipeline occasionally failed owing to default device misconfiguration or model not cached.
OpenAI API access failed because missing or invalid API keys were saved in userdata.
Solution:
Adopted smaller models such as gpt2-medium or gpt2-small for GPU-constrained environments.
Set the device explicitly via device=0 if torch.cuda.is_available() else -1 on model initialization.

Utilized userdata.get('HUGGINGFACEHUB_API_TOKEN') or .env for secret handling and included fallback logic.
Checked model connectivity at runtime prior to running queries and provided useful error messages.

