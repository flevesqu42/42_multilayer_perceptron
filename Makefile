NAME = multilayer_perceptron

EXEC_PATH = target/debug/$(NAME)

RM = rm -f

DEPENDENCIES =	src/main.rs\
				src/maths.rs\
				src/maths/matrix.rs\
				src/analytics.rs\
				src/predict.rs\
				src/learning.rs\
				src/configuration.rs\
				src/dataset/mod.rs\
				src/perceptron/mod.rs\
				src/perceptron/layer.rs\
				src/perceptron/predict.rs\
				src/perceptron/configuration.rs\

all : $(EXEC_PATH) $(NAME)

$(EXEC_PATH) : $(DEPENDENCIES)
	cargo build

$(NAME) :
	ln -sf $(EXEC_PATH) $@

clean :
	$(RM) $(NAME)

fclean : clean
	cargo clean

re : fclean all

.PHONY: all clean fclean re
