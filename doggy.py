# Класс объектов Dog
class Dog:
    # Метод для инициализации объекта внутренними данными
    def __init__(self, petname, temp):
        self.name = petname
        self.temperature = temp

    # Получить состояние
    def status(self):
        print("Имя собаки: ", self.name)
        print("Температура собаки:", self.temperature)
        pass

    # Задать температуру
    def set_temperature(self, temp):
        self.temperature = temp
        pass

    # СОБАКИ МОГУТ ЛАЯТЬ
    def bark(self):
        print("Woof!")
        pass
    pass
