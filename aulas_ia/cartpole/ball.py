import pygame

pygame.init()

class Raquete:
    def __init__(self, posicao, tamanho, tamanho_da_tela):
        self.rect = pygame.Rect((0, 0), tamanho)  # Cria o retângulo
        self.rect.center = posicao  # Coloca o centro dele na posição
        self.tamanho_da_tela = tamanho_da_tela

    def desenhar(self, janela):
        pygame.draw.rect(janela, (255, 255, 255), self.rect)

    def mover(self, distancia):
        self.rect.move_ip(distancia)  # Move o retângulo
        self.rect.clamp_ip((0, 0), self.tamanho_da_tela)  # Não deixa sair da tela

class Jogador:
    def __init__(self, raquete, tecla_baixo, tecla_cima):
        self.raquete = raquete
        self.tecla_baixo = tecla_baixo
        self.tecla_cima = tecla_cima
        self.baixo_apertada = False
        self.cima_apertada = False

    def processar_evento(self, evento):
        # Se o evento for o apertar de uma tecla
        if event.type == pygame.KEYDOWN:
            if event.key == self.tecla_cima:
                self.cima_apertada = True
            elif event.key == self.tecla_baixo:
                self.baixo_apertada = True
        # Se o evento for o soltar de uma tecla
        elif event.type == pygame.KEYUP:
            if event.key == self.tecla_cima:
                self.cima_apertada = False
            elif event.key == self.tecla_baixo:
                self.baixo_apertada = False

    def atualizar(self):
        if self.cima_apertada and not self.baixo_apertada:
            self.raquete.mover((0, -20))  # Atenção: -1 é pra cima
        elif self.baixo_apertada and not self.cima_apertada:
            self.raquete.mover((0, 20))  # Atenção: 1 é pra baixo

    def desenhar_raquete(self, janela):
        self.raquete.desenhar(janela)

    def copiar_rect(self):
        return self.raquete.rect.copy()

class Bolinha:
    def __init__(self, tamanho, velocidade, tamanho_da_tela):
        self.rect = pygame.Rect((0, 0), tamanho)
        self.velocidade = velocidade
        # Salva a largura e altura separadas
        self.largura_tela, self.altura_tela = tamanho_da_tela

        # Centraliza a bolinha
        self.rect.center = (self.largura_tela/2, self.altura_tela/2)

    def desenhar(self, janela):
        pygame.draw.circle(janela, (255, 255, 255), self.rect.center, int(self.rect.height/2))

    def atualizar(self):
        if self.rect.top < 0 or self.rect.bottom > self.altura_tela:
            self.velocidade = (self.velocidade[0], -self.velocidade[1])
        self.rect.move_ip(self.velocidade)

    def checar_colisao(self, lista_rects):
        # checa se algum dos rects colide com a bolinha
        index = self.rect.collidelist(lista_rects)
        if index == -1:
            return
        rect = lista_rects[index]
        self.velocidade = (-(rect.center[0] - self.rect.center[0]) / 2, -(rect.center[1] - self.rect.center[1]) / 2)

    def definir_posicao(self, posicao):
        self.rect.center = posicao

    def definir_velocidade(self, velocidade):
        self.velocidade = velocidade

    def pegar_posicao(self):
        return self.rect.center

janela = pygame.display.set_mode((640, 480))
clock = pygame.time.Clock()
raquete = Raquete((50, 240), (30, 150), (640, 480))
jogador1 = Jogador(raquete, pygame.K_s, pygame.K_w)
jogador2 = Jogador(Raquete((590, 240), (30, 150), (640, 480)), pygame.K_DOWN, pygame.K_UP)
bolinha = Bolinha((20, 20), (10, 10), (640, 480))

while True:
    for event in pygame.event.get():
        # Processa os eventos
        jogador1.processar_evento(event)
        jogador2.processar_evento(event)
        if event.type == pygame.QUIT:
            # Termina o programa
            quit()

    if not (640 > bolinha.pegar_posicao()[0] > 0):  # Saiu da tela
        if bolinha.pegar_posicao()[0] < 320:
            # Se saiu pra esquerda, vai pra direita
            bolinha.definir_velocidade((10, 0))

        else:
            # Se saiu pra direita, vai pra esquerda
            bolinha.definir_velocidade((-10, 0))

        bolinha.definir_posicao((320, 240))
    janela.fill((0, 0, 0))  # Deixa a janela preta
    jogador1.atualizar()  # Atualiza o jogador
    jogador1.desenhar_raquete(janela)
    jogador2.atualizar()  # Atualiza o jogador
    jogador2.desenhar_raquete(janela)
    bolinha.checar_colisao([jogador1.copiar_rect(), jogador2.copiar_rect()])
    bolinha.atualizar()
    bolinha.desenhar(janela)
    #raquete.desenhar(janela)
    pygame.display.flip()  # Atualiza a janela com as mudanças
    clock.tick(30)