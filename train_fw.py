import torch
from torch import Tensor
from data import MINDatasetSampler
from model2 import Conv4, Conv6
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip

def euclidean_distance(prototype_features: Tensor, query_features: Tensor) -> Tensor:
    nc, feature_dim = prototype_features.size()
    nq, feature_dim = query_features.size()
    prototype_features = prototype_features.unsqueeze(0).expand(nq, nc, feature_dim)
    query_features = query_features.unsqueeze(1).expand(nq, nc, feature_dim)
    return torch.sum(torch.square(prototype_features - query_features), dim=2)

def train_episode(feature_extractor: torch.nn.Module,
                  data_sampler: MINDatasetSampler, 
                  num_class: int, 
                  num_support: int, 
                  num_query: int) -> Tensor:
    criterion = torch.nn.CrossEntropyLoss()
    support_set_list, query_set_list = data_sampler.random_sample_classes(num_class, num_support, num_query)
    prototype_list = []
    for support_set  in support_set_list:
        support_features = feature_extractor(support_set)
        prototype = torch.sum(support_features, dim=0) / num_support
        prototype_list.append(prototype)
    prototype_features = torch.stack(prototype_list, dim=0)

    loss = 0
    for index, query_set in enumerate(query_set_list):
        query_features = feature_extractor(query_set)
        distance = euclidean_distance(prototype_features,query_features)
        loss += criterion(-distance, torch.tensor(num_query*[index], device="cuda")) # scores = -distance
    return loss

def train(num_episodes : int,
          learning_rate : float,
          feature_extractor: torch.nn.Module,
          train_data_sampler: MINDatasetSampler,
          val_data_sampler :  MINDatasetSampler,
          num_class: int, 
          num_support: int, 
          num_query: int) -> None:
    
    optimizer = torch.optim.Adam(params=feature_extractor.parameters(), lr=learning_rate)
    best_val_loss = float("inf")
    for i in range(num_episodes):
        optimizer.zero_grad()
        episode_loss = train_episode(feature_extractor, train_data_sampler, num_class, num_support, num_query)
        episode_loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            with torch.no_grad():
                val_loss = train_episode(feature_extractor, val_data_sampler, num_class, num_support, num_query)
                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    print("Current best vall loss:", best_val_loss.item(), "at episode", i)
                    torch.save(feature_extractor.state_dict(), "fe_best_val.pt".format(i))

    torch.save(feature_extractor.state_dict(), "fe_last.pt")
        


    
if __name__ == "__main__":
    feature_extractor = Conv4(inplanes=3).to("cuda")
    transforms = Compose([
        RandomResizedCrop(size=(84,84)),
        RandomHorizontalFlip()
    ])
    train_data_sampler = MINDatasetSampler("images/train", transform=transforms, read_all=True, device="cuda")
    val_data_sampler = MINDatasetSampler("images/val", transform=transforms, read_all=True, device="cuda")
    num_episodes = 1200
    learning_rate = 1e-3
    num_class = 5
    num_support = 5
    num_query = 16
    train(num_episodes, learning_rate, feature_extractor, train_data_sampler, val_data_sampler, num_class, num_support, num_query)

