subroutine get_dos(nk, nb, weights, eigenvalues, sigma, nedos, energies, dos)

    !Description:
    !input parameters:
    !   nk (int*4): the number of k points
    !   nb (int*4): the number of bands 
    !   weight(nk) (real*8): the weights of kpoints 
    !   sigma (real*8): the width for the gaussian function
    !   n (int*4): the size of eigenvalues
    !   eigenvalues(n) (real*8): the eigen values 
    !   nedos (int*4): the number of the energy points for dos
    !
    !output parameter:
    !   energies(nedos) (real*8): the energies of dos
    !   dos(nedos) (real*8): the DOS

    implicit none
    integer(kind=4), intent(in) :: nk, nb, nedos
    real(kind=8), intent(in) :: sigma
    real(kind=8), intent(in), dimension(nk, nb) :: eigenvalues
    real(kind=8), intent(in), dimension(nk) :: weights
    real(kind=8), intent(out), dimension(nedos) :: energies, dos
    real(kind=8) :: pi
    real(kind=8) :: emin, emax, de, e, summ, tmp, e_bk
    integer(kind=4) :: i,ik, ib, ind

    pi = 3.141592653589793
    emin = minval(eigenvalues) - 0.01
    emax = maxval(eigenvalues) + 0.01
    de = (emax - emin)/nedos
    do i=1, nedos
        e = emin + i * de
        energies(i) = e
        summ = 0.
        do ik=1, nk
            ind = minloc(eigenvalues(ik,:), dim=1, mask=(eigenvalues(ik,:)>=e-100*sigma))
 band_loop: do ib=ind, nb
                e_bk = eigenvalues(ik,ib)
                if (e_bk >= e+ 100*sigma) then
                    exit band_loop
                else
                    tmp = 1./( sqrt(2.*pi) * sigma ) * exp( -(e-e_bk)**2 / (2.*sigma**2))
                end if
                summ = summ + tmp*weights(ik)
            end do band_loop
        end do
        dos(i) = summ
    end do
    dos = dos/sum(weights)
end subroutine get_dos

subroutine get_dos_fix_elim(emin, emax, nk, nb, weights, eigenvalues, sigma, nedos, energies, dos)

    !Description:
    !input parameters:
    !   nk (int*4): the number of k points
    !   nb (int*4): the number of bands 
    !   weight(nk) (real*8): the weights of kpoints 
    !   sigma (real*8): the width for the gaussian function
    !   n (int*4): the size of eigenvalues
    !   eigenvalues(n) (real*8): the eigen values 
    !   nedos (int*4): the number of the energy points for dos
    !
    !output parameter:
    !   energies(nedos) (real*8): the energies of dos
    !   dos(nedos) (real*8): the DOS

    implicit none
    integer(kind=4), intent(in) :: nk, nb, nedos
    real(kind=8), intent(in) :: sigma, emin, emax
    real(kind=8), intent(in), dimension(nk, nb) :: eigenvalues
    real(kind=8), intent(in), dimension(nk) :: weights
    real(kind=8), intent(out), dimension(nedos) :: energies, dos
    real(kind=8) :: pi
    real(kind=8) :: de, e, summ, tmp, e_bk
    integer(kind=4) :: i,ik, ib, ind

    pi = 3.141592653589793
    de = (emax - emin)/nedos
    do i=1, nedos
        e = emin + i * de
        energies(i) = e
        summ = 0.
        do ik=1, nk
            ind = minloc(eigenvalues(ik,:), dim=1, mask=(eigenvalues(ik,:)>=e-100*sigma))
 band_loop: do ib=ind, nb
                e_bk = eigenvalues(ik,ib)
                if (e_bk >= e+ 100*sigma) then
                    exit band_loop
                else
                    tmp = 1./( sqrt(2.*pi) * sigma ) * exp( -(e-e_bk)**2 / (2.*sigma**2))
                end if
                summ = summ + tmp*weights(ik)
            end do band_loop
        end do
        dos(i) = summ
    end do
    dos = dos/sum(weights)
end subroutine get_dos_fix_elim
