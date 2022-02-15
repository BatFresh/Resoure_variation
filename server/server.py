import flwr as fl


if __name__ == "__main__":

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,
        fraction_eval=0.5,
        
    )

    # Start server
    fl.server.start_server(
        server_address="10.21.23.159:8088",
        config={"num_rounds": 10},
        strategy=strategy,
    )
