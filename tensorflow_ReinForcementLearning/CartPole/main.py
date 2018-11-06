import CartPole

CartPole.CartPole(Train=False, epoch=1000, game_step=1000, gradient_update=10, learning_rate=0.01,
         discount_factor=0.95, save_weight=100, save_path="CartPole", rendering=False).Graph()
