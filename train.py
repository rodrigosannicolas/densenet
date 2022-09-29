def train_epoch(training_model, loader, criterion, optim):
    training_model.train()

    epoch_loss = 0.0

    all_labels = []
    all_predictions = []
    
    for images, labels in loader:
      all_labels.extend(labels.numpy())  

      optim.zero_grad()

      predictions = training_model(images.to(device))
      all_predictions.extend(torch.argmax(predictions, dim=1).cpu().numpy())

      loss = criterion(predictions, labels.to(device))
      
      loss.backward()
      optim.step()

      epoch_loss += loss.item()

    accuracy = accuracy_score(all_labels, all_predictions) * 100
    recall = recall_score(all_labels, all_predictions, average="macro") * 100
    precision = precision_score(all_labels, all_predictions, average="macro") * 100
    f1 = f1_score(all_labels, all_predictions, average="macro") * 100

    return accuracy, recall, precision, f1


def validation_epoch(val_model, loader, criterion):
    val_model.eval()

    epoch_loss = 0.0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
        all_labels.extend(labels.numpy())  

        predictions = val_model(images.to(device))
        all_predictions.extend(torch.argmax(predictions, dim=1).cpu().numpy())

        loss = criterion(predictions, labels.to(device))

        epoch_loss += loss.item()

    accuracy = accuracy_score(all_labels, all_predictions) * 100
    recall = recall_score(all_labels, all_predictions, average="macro") * 100
    precision = precision_score(all_labels, all_predictions, average="macro") * 100
    f1 = f1_score(all_labels, all_predictions, average="macro") * 100

    return accuracy, recall, precision, f1


def train_model(model, train_loader, test_loader, criterion, optim, number_epochs):
    train_accuracy_history = []
    train_recall_history = []
    train_precision_history = []
    train_f1_history = []

    test_accuracy_history = []
    test_recall_history = []
    test_precision_history = []
    test_f1_history = []

    for epoch in range(number_epochs):
        start_time = time.time()
        train_accuracy, train_recall, train_precision, train_f1 = train_epoch(model, train_loader, criterion, optimizer)
        
        train_accuracy_history.append(train_accuracy)
        train_recall_history.append(train_recall)
        train_precision_history.append(train_precision)
        train_f1_history.append(train_f1)
        
        print(f"Training epoch {epoch + 1} | Accuracy {train_accuracy:.2f} | Recall {train_recall:.2f}% | Precision {train_precision:.2f}% | F1 {train_f1:.2f}% | Time {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        test_accuracy, test_recall, test_precision, test_f1 = validation_epoch(model, test_loader, criterion)
        
        test_accuracy_history.append(test_accuracy)
        test_recall_history.append(test_recall)
        test_precision_history.append(test_precision)
        test_f1_history.append(test_f1)
        
        print(f"Validation epoch {epoch + 1} | Accuracy {test_accuracy:.2f} | Recall {test_recall:.2f}% | Precision {test_precision:.2f}% | F1 {test_f1:.2f}% | Time {time.time() - start_time:.2f} seconds")

    return train_accuracy_history, train_recall_history, train_precision_history, train_f1_history, test_accuracy_history, test_recall_history, test_precision_history, test_f1_history,

