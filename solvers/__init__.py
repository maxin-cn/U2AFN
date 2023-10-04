from .InpaintSolver import InpaintSolver

def create_solver(opt):
    if opt['mode'] == 'inpainting':
        solver = InpaintSolver(opt)
    else:
        raise NotImplementedError

    return solver