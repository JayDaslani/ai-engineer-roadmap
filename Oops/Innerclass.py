class Car:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model
        self.engine = self.Engine()

    class Engine:
        def __init__(self):
            self.status = "off"

        def start(self):
            self.status = "Running"
            print('Engine Started')

        def stop(self):
            self.status = "off"
            print('Engine Stopped')

    def drive(self):
        if self.engine.status == "Running":
            print(f"Driving the {self.brand} {self.model}")
        else:
            print("Start the engine first!")
            

car = Car('Ford', 'mustang')
car.drive()
car.engine.start()
car.drive()
car.engine.stop()
car.drive()
