import os
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)



class ClinicalFeature:
    def __init__(self, max_length=512):
        # self.tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        model_path = os.path.join(script_dir, './Bio_ClinicalBERT')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.max_length = max_length

        self.demographic_abcd_map = {
            'gender': {1: 'male', 2: 'female', 3: 'male', 4: 'female'},
            'race': {1: 'white', 2: 'black', 3: 'Hispanic', 4: 'Asian', 5: 'other'},
        }

    def create_abcd_template(self, clinic_factors):
        # number -> text
        texts = []

        for i in range(clinic_factors.shape[0]):
            age = clinic_factors[i, 0].item() / 12.0  # month->year
            gender = clinic_factors[i, 1].item()
            race = clinic_factors[i, 2].item()

            trauma_exposure = clinic_factors[i, 3].item()
            med_rat = clinic_factors[i, 4].item() / 4.0
            sleep_disturb = clinic_factors[i, 5].item()

            parent_anxious_depress = clinic_factors[i, 6].item()
            parent_withdrawn = clinic_factors[i, 7].item()
            parent_somatic_complaints = clinic_factors[i, 8].item()
            parent_thought = clinic_factors[i, 9].item()
            parent_attention = clinic_factors[i, 10].item()
            parent_aggressive = clinic_factors[i, 11].item()
            parent_rulebreaking = clinic_factors[i, 12].item()

            parent_edu = clinic_factors[i, 13].item()
            family_conflict = clinic_factors[i, 14].item()
            prosocial_behavior = clinic_factors[i, 15].item()

            text = [
                f"Assess mental health risk with this data:",
                f"[age] {age:.2f} years",
                f"[gender] {self.demographic_abcd_map['gender'].get(gender, 'unknown')}",
                f"[race] {self.demographic_abcd_map['race'].get(race, 'unknown')}",
                f"[parent education] {parent_edu:.2f} years",

                f"[trauma exposure severity] negative normalized score {trauma_exposure:.2f}",
                f"[medical rating severity] negative normalized score {med_rat:.2f}",
                f"[sleep disturbance] negative normalized score {sleep_disturb:.2f}",

                f"[Parent anxiety or depression] negative normalized score {parent_anxious_depress:.2f}",
                f"[parent withdrawn] negative normalized score {parent_withdrawn:.2f}",
                f"[parent somatic complaints] negative normalized score {parent_somatic_complaints:.2f}",
                f"[parent thought problem] negative normalized score {parent_thought:.2f}",
                f"[parent attention problem] negative normalized score {parent_attention:.2f}",
                f"[parent aggressive behavior] negative normalized score {parent_aggressive:.2f}",
                f"[parent rule-breaking behavior] negative normalized score {parent_rulebreaking:.2f}",

                f"[family conflict] negative normalized score {family_conflict:.2f}",
                f"[prosocial behavior] positive normalized score {prosocial_behavior:.2f}",
            ]

            # print('text:', text)
            full_text = ' '.join(text)
            texts.append(full_text)

        return texts

    def convert_batch(self, clinic_factors, data_name, sub_name, device):
        # number -> text
        if data_name == 'ABCD':
            batch_texts = self.create_abcd_template(clinic_factors)

        # text -> tokens
        batch_tokens = self.tokenizer(
            batch_texts,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        ).to(device)

        return batch_tokens


class ClinicClassifier(nn.Module):
    # def __init__(self, bert_model_name='emilyalsentzer/Bio_ClinicalBERT', num_classes=1):
    def __init__(self, bert_model_name=os.path.join(script_dir, './Bio_ClinicalBERT'), num_classes=1):
        super().__init__()
        self.converter = ClinicalFeature()

        self.biobert = AutoModel.from_pretrained(bert_model_name)
        for param in self.biobert.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(self.biobert.config.hidden_size, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, clinic_factors, data_name, sub_name, device):
        clinic_factors = clinic_factors.to(device)
        clinic_tokens = self.converter.convert_batch(clinic_factors, data_name, sub_name, device)

        input_ids = clinic_tokens['input_ids'].to(device)
        attention_mask = clinic_tokens['attention_mask'].to(device)

        with torch.no_grad():
            outputs = self.biobert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch, 768]

        x1 = self.fc1(cls_embedding)  # [batch, 256]
        x = self.relu(x1)
        out = self.fc2(x)  # [batch, 1]

        return out, x1