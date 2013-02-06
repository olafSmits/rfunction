module cnct

contains

subroutine genlda(lda, par, gpar, nterms, mpar)
implicit none
integer nterms, mpar
integer, parameter :: k1 = SELECTED_INT_KIND(16)
integer, parameter :: k2 = SELECTED_REAL_KIND(10,300)
REAL(kind = k2) par(0:mpar-1), gpar(0:mpar-1)
REAL(kind = k2) lda(0:nterms-1), ts(0:nterms-1), par_power(0:mpar-1)
REAL(kind = k2) tinynumber
integer nstops(0:9)
integer i, j, m

!f2py intent(in,out) lda
lda(:) = 0
tinynumber =  REAL(10., kind = k2)**(-REAL(300,kind = k2))
!write(*,*) tinynumber
par_power = par
lda(0) = 1
ts(0) = 0

do i = 1, nterms - 1
ts(i) = DOT_PRODUCT(gpar, par_power)
par_power = par_power * par
end do

do i = 1, nterms - 1
lda(i) = DOT_PRODUCT(lda(i-1:0:-1),ts(1:i)) / REAL(i,kind = k2)
if (mod(i, 100000) == 0) then
write(*,*) 'Term: ', i
write(*,*) 'Value: ', lda(i)
end if
if (lda(i) < tinynumber) then
write(*,*) 'Smallest precision reached. Exiting loop.'
exit
end if
end do

return
end subroutine genlda


end module cnct