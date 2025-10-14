import json
import os
import random

from PIL import Image
from torch.utils.data import Dataset


class COCODataSet(Dataset):
    def __init__(self, data_path, trans):
        self.data_path = data_path
        self.trans = trans

        img_files = os.listdir(self.data_path)
        random.shuffle(img_files)
        self.img_files = img_files

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_file = self.img_files[index]
        img_id = int(img_file.split(".jpg")[0][-6:])

        image = Image.open(os.path.join(self.data_path, img_file)).convert("RGB")
        image = self.trans(image)

        return {"img_id": img_id, "image": image}


class POPEDataSet(Dataset):
    def __init__(self, pope_path, data_path, trans):
        self.pope_path = pope_path
        self.data_path = data_path
        # print(data_path)
        self.trans = trans

        image_list, query_list, label_list = [], [], []


        for q in open(pope_path, 'r'):
            line = json.loads(q)
            image_list.append(line['image'])
            query_list.append(line['text'])
            label_list.append(line['label'])

        for i in range(len(label_list)):
            if label_list[i] == 'no':
                label_list[i] = 0
            else:
                label_list[i] = 1

        assert len(image_list) == len(query_list)
        assert len(image_list) == len(label_list)

        self.image_list = image_list
        self.query_list = query_list
        self.label_list = label_list

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.image_list[index])
        raw_image = Image.open(image_path).convert("RGB")
        # image = self.trans(raw_image)
        image = raw_image
        query = self.query_list[index]
        label = self.label_list[index]
        return {"image": image, "query": query, "label": label}

class POPEDataSetEval(Dataset):
    def __init__(self, pope_path, data_path, trans):
        self.pope_path = pope_path
        self.data_path = data_path
        # print(data_path)
        self.trans = trans

        image_list, query_list, label_list, question_id = [], [], [], []


        for q in open(pope_path, 'r'):
            line = json.loads(q)
            image_list.append(line['image'])
            query_list.append(line['text'])
            label_list.append(line['label'])
            question_id.append(line['question_id'])

        # for i in range(len(label_list)):
        #     if label_list[i] == 'no':
        #         label_list[i] = 0
        #     else:
        #         label_list[i] = 1

        assert len(image_list) == len(query_list)
        assert len(image_list) == len(label_list)

        self.image_list = image_list
        self.query_list = query_list
        self.label_list = label_list
        self.question_id = question_id

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.image_list[index])
        raw_image = Image.open(image_path).convert("RGB")
        # image = self.trans(raw_image)
        image = raw_image
        query = self.query_list[index]
        label = self.label_list[index]
        question_id = self.question_id[index]
        return {"image": image, "query": query, "label": label, "question_id": question_id}

class POPEDataSetEvalWrong(Dataset):
    def __init__(self, pope_path, data_path, trans, filter_path=None):
        self.pope_path = pope_path
        self.data_path = data_path
        self.trans = trans

        # 1. 필터링할 question_id 읽기
        if filter_path is not None:
            with open(filter_path, 'r') as f:
                filter_data = [json.loads(line) for line in f]
            self.filter_ids = set(item["question_id"] for item in filter_data)
        else:
            self.filter_ids = None  # 필터 없이 전체 사용

        image_list, query_list, label_list, question_id = [], [], [], []

        # 2. 전체 pope_path jsonl 파일 읽기
        for q in open(pope_path, 'r'):
            line = json.loads(q)
            qid = line['question_id']

            # 3. 만약 filter_ids가 주어졌다면 해당하는 것만 추가
            if (self.filter_ids is None) or (qid in self.filter_ids):
                image_list.append(line['image'])
                query_list.append(line['text'])
                label_list.append(line['label'])
                question_id.append(qid)

        assert len(image_list) == len(query_list) == len(label_list) == len(question_id)

        self.image_list = image_list
        self.query_list = query_list
        self.label_list = label_list
        self.question_id = question_id

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.image_list[index])
        raw_image = Image.open(image_path).convert("RGB")
        image = raw_image  # 변환 적용 안 하는 걸로 유지
        query = self.query_list[index]
        label = self.label_list[index]
        question_id = self.question_id[index]
        return {"image": image, "query": query, "label": label, "question_id": question_id}


class POPEDataSetInterpret(Dataset):
    def __init__(self, pope_path, data_path, trans):
        self.pope_path = pope_path
        self.data_path = data_path
        # print(data_path)
        self.trans = trans

        image_list, query_list, label_list, question_id = [], [], [], []


        for q in open(pope_path, 'r'):
            line = json.loads(q)
            image_list.append(line['image'])
            query_list.append(line['text'])
            label_list.append(line['label'])
            question_id.append(line['question_id'])

        # for i in range(len(label_list)):
        #     if label_list[i] == 'no':
        #         label_list[i] = 0
        #     else:
        #         label_list[i] = 1

        assert len(image_list) == len(query_list)
        assert len(image_list) == len(label_list)

        self.image_list = image_list
        self.query_list = query_list
        self.label_list = label_list
        self.question_id = question_id

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.image_list[index])
        raw_image = Image.open(image_path).convert("RGB")
        # image = self.trans(raw_image)
        image = raw_image
        query = self.query_list[index]
        label = self.label_list[index]
        question_id = self.question_id[index]
        return {"image": image, "query": query, "label": label, "question_id": question_id}