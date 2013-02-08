module cnct

contains

subroutine genlda(lda, ts, nterms, ndist)
implicit none
integer nterms, ndist
integer, parameter :: k1 = SELECTED_INT_KIND(16)
integer, parameter :: k2 = SELECTED_REAL_KIND(10,300)
REAL(kind = k2) tinynumber
REAL(kind = k2) lda(0:nterms-1, 0:ndist-1), ts(1:nterms-1, 0:ndist-1)
integer i, j

!f2py intent(in,out) lda
tinynumber =  REAL(10., kind = k2)**(-REAL(300,kind = k2))
do j = 0, ndist -1
do i = 1, nterms - 1
lda(i, j) = DOT_PRODUCT(lda(i-1:0:-1, j),ts(1:i, j)) / REAL(i,kind = k2)
if (lda(i, j) < tinynumber) then
write(*,*) 'Smallest precision reached'
exit
end if
end do

end do

return
end subroutine genlda



subroutine levin_acceleration(terms, beta, nterms, mvolt, kdist)
implicit none
integer, parameter :: k1 = SELECTED_INT_KIND(16)
integer, parameter :: k2 = SELECTED_REAL_KIND(10,300)
integer nterms, mvolt, maxA, kdist
complex(kind = k2) terms(0:nterms-1, 0:mvolt-1, 0:kdist-1)
complex(kind = k2) tempstorage(0:nterms-2, 0:mvolt-1, 0:kdist-1), cumsum(0:mvolt-1, 0:kdist-1)
complex(kind = k2) numerator(0:mvolt-1, 0:kdist-1), denominator(0:mvolt-1, 0:kdist-1), beta
complex(kind = k2) ONE
integer i
!f2py intent(in,out) terms

ONE = CMPLX(1, kind = 8)

!define differences
do i = 0,nterms-2
tempstorage(i,:,:) = ONE / terms(i+1,:,:)
end do
call recursivegenerator(tempstorage, nterms, mvolt, kdist, beta)
denominator(:,:) = tempstorage(0,:,:)


!define partial sums / differences
tempstorage(0,:,:) = ONE
cumsum(:,:) = terms(0,:,:)
do i = 0,nterms-2
tempstorage(i,:,:) = cumsum(:,:) / terms(i+1,:,:)
cumsum(:,:) = cumsum(:,:) + terms(i+1,:,:)
end do


call recursivegenerator(tempstorage, nterms, mvolt, kdist, beta)
numerator(:,:) = tempstorage(0,:,:)

terms(0,:,:) = numerator(:,:)/denominator(:,:)

return
end subroutine levin_acceleration


!this routine generates the desired numerator / denominator of the Levin transform
!recursively. 
subroutine recursivegenerator(s, nterms, mvolt, kdist, beta)
implicit none
integer, parameter :: k1 = SELECTED_INT_KIND(16)
integer, parameter :: k2 = SELECTED_REAL_KIND(10,300)
integer nterms, mvolt, kdist
complex(kind = k2) s(0:nterms-2,0:mvolt-1, 0:kdist-1), beta, tau, bu, bd
integer n, k

!f2py intent(in,out) s

! bn = (beta+n-1)*(beta+n-2)
! do j = 1, nterms-1
! bj =(beta+n+j-2.) *(beta+n+j-3.)
! tau = bn / bj
! s(n-j,:) = s(n-j+1,:) - tau* s(n-j,:)
! end do
! return

do k=1, nterms-1
do n=0, nterms - 2 - k
bu = (beta+n+k)*(beta+n+k-1)
bd = (beta+n+2*k)*(beta+n+2*k-1)
tau = bu / bd
s(n,:,:) = s(n+1,:,:) - tau * s(n,:,:)
end do
end do
return
end subroutine recursivegenerator

end module cnct