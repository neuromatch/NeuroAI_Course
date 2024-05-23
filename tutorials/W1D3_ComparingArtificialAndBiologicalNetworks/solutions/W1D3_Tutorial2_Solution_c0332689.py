
args = build_args()
torch.manual_seed(args.seed+1)
args.epochs = 1
#build_model
model_new_seed = Net().to(device)

optimizer = optim.Adadelta(model_new_seed.parameters(), lr=args.lr)
train_model(args, model_new_seed, optimizer)