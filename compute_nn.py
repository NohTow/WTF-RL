
import faiss
import numpy as np
from tqdm import tqdm

with open("embeddings/caption_embeddings_train.npy", "rb") as fp:   # Unpickling
    caption_embeds = np.load(fp)

with open("embeddings/image_embeddings_train.npy", "rb") as fp:   # Unpickling
    img_embeds = np.load(fp)


index = faiss.read_index("embeddings/IVFFlatText_train.faiss")
index.nprobe = 128


nearest_neighbors = np.zeros((len(img_embeds), 5), dtype=int)
batch_size = 10000
u = 0
pbar=tqdm(total=len(img_embeds))
while(u < len(img_embeds)):
    query = img_embeds[u:u+batch_size]
    D, I = index.search(query, 6)

    for i, elem in enumerate(I):
        if(i in elem):
            nearest_neighbors[i+u] = [indice for indice in elem if (indice != i)]
        else:
            nearest_neighbors[i+u] = [indice for indice in elem[:5]]
        # print([indice for indice in elem if (indice != i)])
    u += batch_size
    pbar.update(batch_size)
if(u != len(img_embeds)):
   
    query = img_embeds[u:]
    D, I = index.search(query, 6)
    for i, elem in enumerate(I):
        if(i in elem):
            nearest_neighbors[i+u] = [indice for indice in elem if (indice != i)]
        else:
            nearest_neighbors[i+u] = [indice for indice in elem[:5]]
        # print([indice for indice in elem if (indice != i)])


with open("embeddings/nn_train.npy", 'wb') as f:
    np.save(f, nearest_neighbors, allow_pickle=False)
