import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import wandb
import pickle

from model import SMARTEModel
from loss import SetLoss
from utils import collate_fn, formulate_gold, generate_triple
from metric import metric, t_test


class REDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Trainer:
    def __init__(self, args, model, train_dataset, val_dataset):
        self.args = args
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model_save_path = args.model_save_path
        self.best_val_f1 = 0
        
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        component = ['encoder', 'decoder']
        grouped_params = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and component[0] in n],
                'weight_decay': args.weight_decay,
                'lr': args.encoder_lr
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and component[0] in n],
                'weight_decay': 0.0,
                'lr': args.encoder_lr
            },
            {
                'params': [p for n, p in self.model.named_parameters() if
                           not any(nd in n for nd in no_decay) and component[1] in n],
                'weight_decay': args.weight_decay,
                'lr': args.decoder_lr
            },
            {
                'params': [p for n, p in self.model.named_parameters() if
                           any(nd in n for nd in no_decay) and component[1] in n],
                'weight_decay': 0.0,
                'lr': args.decoder_lr
            }
        ]
        
        self.optimizer = torch.optim.AdamW(grouped_params)
        # Include Warm-up steps to stabilize early training
        total_steps = len(train_dataset) * args.epochs
        warmup_steps = int(total_steps * 0.1)  # 10% of total steps for warm-up
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps,
                                                         num_training_steps=total_steps)
        

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc="Training"):
            self.optimizer.zero_grad()
            outputs = self.model(batch['input_ids'].to(self.args.device), 
                                 batch['attention_mask'].to(self.args.device))
            loss = SetLoss(outputs=outputs, targets=batch['targets'], num_classes=self.args.num_classes)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            self.model.zero_grad()
            self.scheduler.step()  # Update the learning rate
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def eval_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0
        prediction, gold = {}, {}
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                outputs = self.model(batch['input_ids'].to(self.args.device), 
                                     batch['attention_mask'].to(self.args.device))
                loss = SetLoss(outputs=outputs, targets=batch['targets'], num_classes=self.args.num_classes)
                gold.update(formulate_gold(batch['targets'], batch['info']))
                pred_triple = generate_triple(outputs, batch['info'], self.args, self.args.num_classes)
                prediction.update(pred_triple)
                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        t_test(prediction, gold)
        eval_scores = metric(prediction, gold)

        return avg_loss, eval_scores

    def train(self):
        for epoch in range(self.args.epochs):
            train_dataloader = self.train_dataset
            val_dataloader = self.val_dataset
            avg_train_loss = self.train_epoch(train_dataloader, epoch)
            avg_val_loss, eval_scores = self.eval_epoch(val_dataloader)
            wandb.log({"train_loss": avg_train_loss, 
                       "val_loss": avg_val_loss, 
                       "val_precision": eval_scores['precision'], 
                       "val_recall": eval_scores['recall'],
                       "val_f1": eval_scores['f1'],
                       "epoch": epoch})

            print(f"Epoch {epoch + 1}: Training Loss {avg_train_loss}, Validation Loss {avg_val_loss}")
            
            if eval_scores['f1'] > self.best_val_f1:
                self.best_val_f1 = eval_scores['f1']
                print(f"New best validation F1 Score {self.best_val_f1}. Saving model and tokenizer.")
                #torch.save(self.model.state_dict(), self.model_save_path)


    def evaluate(self):
        val_dataloader = self.val_dataset
        avg_val_loss, eval_scores, num_metric_scores, overlap_metric_scores = self.eval_epoch(val_dataloader)
        print(f"Validation Loss {avg_val_loss}")
        print(f"Val precision: {eval_scores['precision']}, Val recall: {eval_scores['recall']}, Val f1: {eval_scores['f1']}")
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="WebNLG-Exact")
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)
    parser.add_argument("--gpu", default="cuda:0", type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--epochs", default=340, type=int)
    
    # SMARTE Parameters:
    parser.add_argument("--num_generated_triples", default=15, type=int)
    parser.add_argument('--encoder_lr', type=float, default=2e-5)
    parser.add_argument('--decoder_lr', type=float, default=6e-5)
    parser.add_argument('--mesh_lr', type=float, default=6)
    parser.add_argument('--n_mesh_iters', type=int, default=4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=2.5)
    parser.add_argument('--num_iterations', default=6, type=int)
    parser.add_argument('--slot_dropout', default=0.2, type=float)
    
    # Others
    parser.add_argument("--model_save_path", type=str)
    parser.add_argument("--model_load_path", type=str)
    parser.add_argument("--name", type=str, default="SMARTE42")
    parser.add_argument("--project", type=str, default="WebNLGExact-ACL")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    
    # Inclusive of NA class
    if args.data_type == "NYT-Exact":
        args.num_classes = 25
    elif args.data_type == "WebNLG-Exact":
        args.num_classes = 212
    else:
        print("Not Supported Dataset, Terminating Programme")
        return None

    args.model_save_path = f'./best_f1_{args.data_type}_{args.seed}.pth'
    #args.model_load_path = f'./best_f1_{args.data_type}_{args.seed}.pth'
    
    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    wandb.init(
        name=args.name,
        project=args.project,
        notes="None",
        #mode="disabled",
    )
    wandb.config.update(args)

    # Training and Validation Set
    with open(f"./processed/{args.data_type}/traindev_features.pkl", 'rb') as f:
        re_training = pickle.load(f)
    with open(f"./processed/{args.data_type}/test_features.pkl", 'rb') as f:
        re_testing = pickle.load(f)

    training_dataset = REDataset(re_training)
    training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    testing_dataset = REDataset(re_testing)
    testing_dataloader = DataLoader(testing_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = SMARTEModel(args)
    model.to(device)
    
    trainer = Trainer(args=args,
                      model=model,
                      train_dataset=training_dataloader,
                      val_dataset=testing_dataloader)

    if args.model_load_path:
        model.load_state_dict(torch.load(args.model_load_path, map_location=device))
        trainer.evaluate()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
