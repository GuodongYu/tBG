subroutine get_EBS(ene, nk, nb, eigvals, PMK, sigma, delta_ene, As)
    use omp_lib
    implicit none
    integer(kind=4), intent(in) :: nk, nb
    real(kind=8), intent(in) :: ene
    real(kind=8), intent(in) :: sigma, delta_ene
    real(kind=8), intent(in) :: eigvals(nk,nb), PMK(nk,nb)

    real(kind=8), intent(out) :: As(nk)
    
    integer(kind=4) :: i,j
    real(kind=8) :: A, pi, outt, t0, t1
    

    pi = 3.141592653589793
    As=0.
    !$OMP PARALLEL
    !$OMP DO PRIVATE(j, outt, A)
    do i=1, nk
        A = 0.
        do j=1, nb
            if (ene - eigvals(i,j) > 50*sigma) then
                cycle
            else if (eigvals(i,j)-ene > 50*sigma) then
                exit
            else
                call integ(ene, eigvals(i,j), sigma, delta_ene, outt)
                A =  A + PMK(i, j) * outt
            end if
        end do
        As(i) = A
    end do
    !$OMP END DO
    !$OMP END PARALLEL
end subroutine get_EBS

subroutine get_EBS_from_f(ene, nk, nb, eigvals_f, PMK_f, sigma, delta_ene, As)
    use omp_lib
    implicit none
    integer(kind=4), intent(in) :: nk, nb
    real(kind=8), intent(in) :: ene
    real(kind=8), intent(in) :: sigma, delta_ene
    character(len=20), intent(in) :: eigvals_f, PMK_f
    real(kind=8), intent(out) :: As(nk)
    
    real(kind=8) :: eigvals(nk,nb), PMK(nk,nb)
    integer(kind=4) :: i,j
    real(kind=8) :: A, pi, outt, t0, t1

    open(unit=10, file=eigvals_f, access="stream" , form="unformatted")
    read(10) eigvals
    close(10)
    open(unit=11, file=PMK_f, access="stream" , form="unformatted")
    read(11) PMK
    close(11)
    
    pi = 3.141592653589793
    As=0.
    !$OMP PARALLEL
    !$OMP DO PRIVATE(j, outt, A)
    do i=1, nk
        A = 0.
        do j=1, nb
            if (ene - eigvals(i,j) > 50*sigma) then
                cycle
            else if (eigvals(i,j)-ene > 50*sigma) then
                exit
            else
                call integ(ene, eigvals(i,j), sigma, delta_ene, outt)
                A =  A + PMK(i, j) * outt
            end if
        end do
        As(i) = A
    end do
    !$OMP END DO
    !$OMP END PARALLEL
end subroutine get_EBS_from_f

subroutine integ(ene, e0, sigma, delta_ene, outt)
    implicit none
    real(kind=8), intent(in) :: ene, e0, sigma, delta_ene
    real(kind=8), intent(out) :: outt
    
    real(kind=8) :: de, e, pi


    pi = 3.141592653589793
    de = sigma/50
    outt = 0.0
    e = ene-delta_ene/2
    do while (e<=ene+delta_ene/2)
        outt = outt + 1./(sqrt(2.*pi)*sigma) * exp( -(e-e0)**2 / (2.*sigma**2))
        e = e + de
    end do
    outt = outt * de
end subroutine integ

subroutine get_Pk(k, nlayer, layer_npcs, nspecies, norb_species, nsite_uc, &
                  & norb_uc, nsite, norb, vecs, coords, species, pk)
    use omp_lib
    implicit none
    real(kind=8), intent(in) :: k(2), coords(norb, 3)
    complex(kind=8), intent(in) :: vecs(norb, norb)
    integer(kind=4), intent(in) :: nlayer, nspecies, nsite_uc, norb_uc, nsite, norb
    integer(kind=4), intent(in) :: norb_species(nspecies), layer_npcs(nlayer), species(nsite)
    real(kind=8), intent(out) :: pk(norb)

    integer(kind=4) :: i
    real(kind=8) :: pp
    !$OMP PARALLEL
    !$OMP DO PRIVATE(pp)
    do i=1, norb
        call get_Pnk(k, nlayer, layer_npcs, nspecies, norb_species, nsite_uc, &
                     & norb_uc, nsite, norb, vecs(:,i), coords, species, pp)
        pk(i) = pp
    end do  
    !$OMP END DO      
    !$OMP END PARALLEL
end subroutine get_Pk


subroutine get_Pnk(k_cart, nlayer, layer_npcs, nspecies, norb_species, nsite_uc, &
                   & norb_uc, nsite, norb, vec, coords, species, pp)

    !One |nk> state corresponds to one PMK
    !input:
    !    k_cart(2): the k coordinate in cartisian (real*8)
    !    nlayer: the number of layer (int*4)
    !    layer_npcs(nlayer): the number of unit cell for each layer (int*4)
    !    nspecies: the number of the atom type (int*4)
    !    norb_species(nspecies): the number of orbital type on each specie (int*4)
    !    nsite_uc: the number of site in one unit cell (int*4)
    !    norb_uc: the number of orbital in one unit cell (int*4)
    !    nsite: the number of all sites (int*4)
    !    norb: the number of all orbitals (int*4)
    !    vec(norb): the eigen vectors (complex*16)
    !    coords(nsite,3): the coordinates of all sites (real*16)
    !    species(nsite): the species of all sites (int*4), note: labeled by 1,2,...,nspecies
    !          e.g. MoS2, there are three species Mo, S1, and S2, they should be labeled
    !               by 1, 2, 3
    !output:
    !    pp(norb): the PMK value (real*8)

    implicit none
    integer(kind=4), intent(in) :: nlayer, nspecies, nsite_uc, nsite, norb, norb_uc
    integer(kind=4), intent(in) :: layer_npcs(nlayer), species(nsite), norb_species(nspecies)
    real(kind=8), intent(in) :: k_cart(2),  coords(nsite, 3)
    complex(kind=8), intent(in) :: vec(norb)
    real(kind=8), intent(out) :: pp

    complex(kind=8), parameter :: ii = (0., 1.)
    real(kind=8) :: summ
    real(kind=8) :: vec_diff(3)
    integer(kind=4) ::  npc
    integer(kind=4) :: ind_layer, ind_site, i, ind_orb, species_i, m, n
    integer(kind=4) :: ind0_site_layer, ind0_orb_layer, ind1_site_layer, ind1_orb_layer
    integer(kind=4) :: ind0_orb_inuc, ind1_orb_inuc, ind0_orb_insite, ind1_orb_insite
    integer(kind=4) :: orb_count(norb_uc)
    integer(kind=4), allocatable :: inds_orb_coord(:,:), inds_orb_vec(:,:)

    pp = 0.0 
    do ind_layer=1, nlayer
        summ = 0.0
        npc = layer_npcs(ind_layer) ! number of unit cell in current layer

        ! the starting and ending index in coordinate and vector in current layer
        ! ind0 for starting, ind1 for ending
        if(ind_layer .eq. 1) then
            ind0_site_layer = 1
            ind0_orb_layer = 1
        else 
            ind0_site_layer = sum(layer_npcs(1:ind_layer-1))*nsite_uc + 1   
            ind0_orb_layer = sum(layer_npcs(1:ind_layer-1))*norb_uc + 1    
        end if
        ind1_site_layer = ind0_site_layer + npc*nsite_uc -1
        ind1_orb_layer = ind0_orb_layer + npc*norb_uc -1
        ! inds_orb_coord and inds_orb_vec for saving the index for each orbital
        allocate(inds_orb_coord(norb_uc,npc))
        allocate(inds_orb_vec(norb_uc,npc))

        ! classify the indexes according to the orbital 
        orb_count=0
        do ind_site=ind0_site_layer, ind1_site_layer
            species_i = species(ind_site) ! the species
            
            ! get starting and ending index for species_i in unit cell
            if (species_i .eq. 1) then
                ind0_orb_inuc = 1
            else
                ind0_orb_inuc = sum(norb_species(1:species_i-1)) + 1 
            end if
            ind1_orb_inuc = ind0_orb_inuc + norb_species(species_i) - 1

            ! get the starting and ending index of orbital for site in vector array
            if(ind_site .eq. ind0_site_layer) then
                ind0_orb_insite = ind0_orb_layer
                ind1_orb_insite = ind0_orb_layer + norb_species(species_i) -1
            else
                ind0_orb_insite = ind1_orb_insite + 1 
                ind1_orb_insite = ind1_orb_insite + norb_species(species_i)
            end if

            do i=ind0_orb_inuc, ind1_orb_inuc
                orb_count(i) = orb_count(i) + 1
                inds_orb_coord(i,orb_count(i))=ind_site
                inds_orb_vec(i,orb_count(i))= i - ind0_orb_inuc + ind0_orb_insite
            end do
        end do
        do ind_orb=1, norb_uc
            do m=1, npc-1
                do n=m+1, npc
                    ! m != n
                    vec_diff = coords(inds_orb_coord(ind_orb,m),:) - coords(inds_orb_coord(ind_orb,n),:)
                    summ = summ + 2*real(exp(ii*dot_product(k_cart, vec_diff(1:2)))* &
                                      & (conjg(vec(inds_orb_vec(ind_orb,m)))*vec(inds_orb_vec(ind_orb,n))))
                end do
                ! m = n in ( 1 ~ npc-1 )
                summ = summ + real(vec(inds_orb_vec(ind_orb,m)))**2 + aimag(vec(inds_orb_vec(ind_orb,m)))**2 
            end do
            ! m = n = npc
            summ = summ + real(vec(inds_orb_vec(ind_orb,npc)))**2 + aimag(vec(inds_orb_vec(ind_orb,npc)))**2 
            pp = pp + summ / npc
        end do
        deallocate(inds_orb_coord)
        deallocate(inds_orb_vec)
    end do
end subroutine get_Pnk
