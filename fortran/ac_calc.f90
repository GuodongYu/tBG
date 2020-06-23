subroutine AC_calc(nk, n, Jks, weights, vals, vecs, nw, omega_max, energy_cut, delta, omegas, acs)
    !
    !Parameters
    !input:
    !   nk: the number of k points (int*4)
    !   n: the number of orbitals (int*4)
    !   Jks(nk,n,n): the array of Jk operator matrix at all kpoints (complex*16)
    !   vals(nk,n): the eigen values (real*8)
    !   vecs(nk,n): the eigen vectors (complex*16)
    !   nw: the number of omega points (int*4)
    !   omega_max: up to which the ac conductivity will be calculated (real*8)
    !   energy_cut: at specific omega, the transition cb-vb in the range 
    !               [omega-energy_cut, omega+energy_cut] will be considered
    !               having contribution on AC conductivity at omega (real*8)
    !   delta: the delta parameter in Kubo formula for AC (real*8)
    !
    !output:
    !   omegas(nw): the omega points array, at each of which 
    !               the AC will be calculated (real*8)
    !   acs(nw): the calculated AC values at omegas (real*8)
    !
    use blas95    
    implicit none
    integer(kind=4), intent(in) :: n, nk, nw
    integer(kind=4), intent(in) :: weights(nk)
    real(kind=8), intent(in) :: vals(nk, n)
    complex(kind=8), intent(in) :: vecs(nk, n, n), Jks(nk, n, n)
    real(kind=8), intent(in) :: omega_max, energy_cut, delta
    real(kind=8), intent(out) :: omegas(nw), acs(nw)

    complex(kind=8) :: jj = (0., 1.)
    integer(kind=4) :: ind_vbm, ind_cbm, ind_vb, ind_cb
    integer(kind=4) :: n_omega, nk_expand
    complex(kind=8) :: vb_vec(n), cb_vec(n),prob, Jk_cb(n), Jk(n,n)
    real(kind=8) :: vb, cb, omega, f_cb, f_vb, ac, cbm, dw, t0, t1
    integer(kind=4) :: iw, i_weig, ik
    !complex(kind=8), external :: dotc
    Jk_cb = 0.
    dw = omega_max/nw
    
    nk_expand = 0
    do i_weig=1, nk
        nk_expand = nk_expand + weights(i_weig)
    end do
    acs = 0.0
    omegas = 0.0
    ind_vbm = int(n/2)
    ind_cbm = ind_vbm + 1
    f_cb = 0.
    f_vb = 1.
    !!$OMP PARALLEL DO PRIVATE(omega, Jk, ik, ind_vb, vb, vb_vec, ind_cb, cb, cb_vec, prob, ac)
    !!$OMP PARALLEL DO PRIVATE(omega, ik, ind_vb, vb, ind_cb, cb, prob, ac, cb_vec, vb_vec)
    do iw = 2, nw
        call cpu_time(t0)
        print *, 'iw=', iw
        omega = dw * (iw-1)
        omegas(iw) = omega
        ac = 0.0
        do ik=1, nk
            cbm = vals(ik, ind_cbm)
            Jk = Jks(ik,:,:)
            do ind_vb = ind_vbm, 1, -1 
                vb = vals(ik, ind_vb)
                if (vb > cbm - omega + energy_cut) then
                    cycle
                else if (vb < cbm - omega -energy_cut) then
                    exit
                else
                    vb_vec = vecs(ik,:,ind_vb)
                    do ind_cb = ind_cbm, n
                        cb = vals(ik, ind_cb)
                        if (cb <= vb + omega - energy_cut) then
                            cycle
                        else if (cb >= vb + omega + energy_cut) then
                            exit
                        else
                            cb_vec = vecs(ik,:,ind_cb)
                            call hemv(Jk, cb_vec, Jk_cb)
                            !call zhemv('U', n, 1, Jk, n, cb_vec, 1, 0, Jk_cb, 1)
                            prob = dotc(vb_vec, Jk_cb)
                            ac = ac + weights(ik)*real(prob*conjg(prob))*aimag( (f_vb-f_cb) / (vb-cb+omega + delta*jj) )
                            !ac = ac + weights(ik)*real(prob*conjg(prob))*(f_vb-f_cb)*(-delta) / ((vb-cb+omega)**2 + delta**2)
                        end if
                    end do
                end if
            end do
        end do
        ac = -2*ac/nk_expand
        acs(iw) = ac 
        call cpu_time(t1)
        print *, '   ',t1-t0, 's'
    end do
    !!$OMP END PARALLEL DO
end subroutine AC_calc


