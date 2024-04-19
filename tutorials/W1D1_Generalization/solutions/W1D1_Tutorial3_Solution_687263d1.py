
class Renderer(nn.Module):
    def __init__(self, blur_sigma=0.5, epsilon=0., blur_fsize=None, PM=None):
        super().__init__()
        if PM is None:
            PM = Parameters()

        self.painter = Painter(PM)
        self.broaden_and_blur = BroadenAndBlur(blur_sigma, epsilon, blur_fsize, PM)

    def forward(self, drawings, blur_sigma=None, epsilon=None):
        """
        Render each drawing by converting the drawing to image ink
        and then applying broaden & blur filters

        Parameters
        ----------
        drawings : list[list[torch.Tensor]] | list[torch.Tensor]
            Input drawings. Each drawing is a list of tensors
        blur_sigma : float | None
            Sigma parameter for blurring. Only used for adaptive blurring.
            Default 'None' means use the blur_sigma from __init__() call

        Returns
        -------
        pimgs : torch.Tensor
            [n,H,W] Pre-conv image probabilities
        """
        # draw the strokes (this part on cpu)
        if not isinstance(drawings[0], list):
            single = True
            drawings = [drawings]
        else:
            single = False

        pimgs = self.painter(drawings)
        pimgs = self.broaden_and_blur(pimgs, blur_sigma, epsilon) # (n,H,W)

        if single:
            pimgs = pimgs[0]

        return pimgs

class Painter(nn.Module):
    def __init__(self, PM=None):
        super().__init__()
        if PM is None:
            PM = Parameters()
        self.ink_pp = PM.ink_pp
        self.ink_max_dist = PM.ink_max_dist
        self.register_buffer('index_mat',
                             torch.arange(PM.imsize[0]*PM.imsize[1]).view(PM.imsize))
        self.register_buffer('space_flip', torch.tensor([-1.,1.]))
        self.imsize = PM.imsize

    @property
    def device(self):
        return self.index_mat.device

    @property
    def is_cuda(self):
        return self.index_mat.is_cuda

    def space_motor_to_img(self, stk):
        return torch.flip(stk, dims=[-1])*self.space_flip

    def add_stroke(self, pimg, stk):
        stk = self.space_motor_to_img(stk)
        # reduce trajectory to only those points that are in bounds
        out = self.check_bounds(stk) # boolean; shape (neval,)
        ink_off_page = out.any()
        if out.all():
            return pimg, ink_off_page
        stk = stk[~out]

        # compute distance between each trajectory point and the next one
        if stk.shape[0] == 1:
            myink = stk.new_tensor(self.ink_pp)
        else:
            dist = torch.norm(stk[1:] - stk[:-1], dim=-1) # shape (k,)
            dist = dist.clamp(None, self.ink_max_dist)
            dist = torch.cat([dist[:1], dist])
            myink = (self.ink_pp/self.ink_max_dist)*dist # shape (k,)

        # make sure we have the minimum amount of ink, if a particular
        # trajectory is very small
        sumink = torch.sum(myink)
        if sumink < 2.22e-6:
            nink = myink.shape[0]
            myink = (self.ink_pp/nink)*torch.ones_like(myink)
        elif sumink < self.ink_pp:
            myink = (self.ink_pp/sumink)*myink
        assert torch.sum(myink) > (self.ink_pp - 1e-4)

        # share ink with the neighboring 4 pixels
        x = stk[:,0]
        y = stk[:,1]
        xfloor = torch.floor(x).detach()
        yfloor = torch.floor(y).detach()
        xceil = torch.ceil(x).detach()
        yceil = torch.ceil(y).detach()
        x_c_ratio = x - xfloor
        y_c_ratio = y - yfloor
        x_f_ratio = 1 - x_c_ratio
        y_f_ratio = 1 - y_c_ratio
        lind_x = torch.cat([xfloor, xceil, xfloor, xceil])
        lind_y = torch.cat([yfloor, yfloor, yceil, yceil])
        inkval = torch.cat([
            myink*x_f_ratio*y_f_ratio,
            myink*x_c_ratio*y_f_ratio,
            myink*x_f_ratio*y_c_ratio,
            myink*x_c_ratio*y_c_ratio
        ])
        # paint the image
        pimg = self.seqadd(pimg, lind_x, lind_y, inkval)
        return pimg, ink_off_page

    def draw(self, pimg, strokes):
        for stk in strokes:
            pimg, _ = self.add_stroke(pimg, stk)
        return pimg

    def forward(self, drawings):
        assert not self.is_cuda
        drawings = drawings_to_cpu(drawings)
        n = len(drawings)
        pimgs = torch.zeros(n, *self.imsize)
        for i in range(n):
            pimgs[i] = self.draw(pimgs[i], drawings[i])
        return pimgs