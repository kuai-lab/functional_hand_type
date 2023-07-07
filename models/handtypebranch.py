import torch
from torch import nn
class  HandTypeClassificationBranch(nn.Module):
    #To predict hand type
    def __init__(self, num_types, hand_feature_dim):        
        super(HandTypeClassificationBranch, self).__init__()
        self.num_types = num_types # 23
        self.hand_feature_dim = hand_feature_dim # 512
        self.classifier = nn.Linear(self.hand_feature_dim, self.num_types)
        

    def forward(self, hand_feature):        
        out={}
        reg_out = self.classifier(hand_feature)
        _,indices=torch.max(reg_out,1)
        out['reg_outs']=reg_out
        out['pred_labels']=indices
        out['reg_possibilities']=nn.functional.softmax(reg_out,dim=1)
        return out
