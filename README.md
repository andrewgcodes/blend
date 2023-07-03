# blend
<a target="_blank" href="https://colab.research.google.com/github/andrewgcodes/blend/blob/main/blend_song_vectors.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.png" alt="Open In Colab"/>
</a>

Take your two favorite songs and find out what song is a perfect blend (with AI and embeddings)

Made using a subset of the new DISCO-10M dataset (https://huggingface.co/datasets/DISCOX/DISCO-10M)

**Works much much better with the full 10M song dataset, instead of just the 200k subset (50x more songs!). Please expect best results with the full dataset**


For working with full dataset, you will need to build the inverted index and tensors yourself:
```
  from collections import defaultdict
  
  # Create an empty inverted index
  inverted_index = defaultdict(list)
  
  # Populate the inverted index
  for i, song in enumerate(train_ds):
      title_words = song['video_title_youtube'].lower().split()
      for word in title_words:
          inverted_index[word].append(i)
  
  
  # Create the 'allembeddings.pt'
  all_embeddings = torch.tensor([song['audio_embedding_spotify'] for song in train_ds]).cuda()
```
