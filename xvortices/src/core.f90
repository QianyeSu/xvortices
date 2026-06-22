module xvortices_core_mod
    use, intrinsic :: iso_c_binding, only: c_double, c_int, c_ptr, c_f_pointer
    use, intrinsic :: ieee_arithmetic, only: ieee_quiet_nan, ieee_value
    implicit none

contains

subroutine cylind_coords_c(olon, olat, azim, radi, ncenter, nazim, nradi, lons, lats, etas) bind(C, name="xvortices_cylind_coords_c")
    type(c_ptr), value, intent(in) :: olon
    type(c_ptr), value, intent(in) :: olat
    type(c_ptr), value, intent(in) :: azim
    type(c_ptr), value, intent(in) :: radi
    integer(c_int), value, intent(in) :: ncenter
    integer(c_int), value, intent(in) :: nazim
    integer(c_int), value, intent(in) :: nradi
    type(c_ptr), value, intent(in) :: lons
    type(c_ptr), value, intent(in) :: lats
    type(c_ptr), value, intent(in) :: etas

    real(c_double), pointer :: olon_v(:), olat_v(:), azim_v(:), radi_v(:)
    real(c_double), pointer :: lons_v(:), lats_v(:), etas_v(:)
    real(c_double), allocatable :: olon_r_v(:), sin_olat(:), cos_olat(:)
    real(c_double), allocatable :: sin_azim(:), cos_azim(:), sin_radi(:), cos_radi(:)
    real(c_double), allocatable :: dlam_base(:, :)
    real(c_double) :: olon_r, azim_sin, azim_cos, radi_sin, radi_cos
    real(c_double) :: lats_r, dlam_r, eta_r, arg
    real(c_double) :: pi, deg2rad, rad2deg
    integer :: ic, ia, ir, idx, total

    call c_f_pointer(olon, olon_v, [int(ncenter)])
    call c_f_pointer(olat, olat_v, [int(ncenter)])
    call c_f_pointer(azim, azim_v, [int(nazim)])
    call c_f_pointer(radi, radi_v, [int(nradi)])
    total = int(ncenter) * int(nradi) * int(nazim)
    call c_f_pointer(lons, lons_v, [total])
    call c_f_pointer(lats, lats_v, [total])
    call c_f_pointer(etas, etas_v, [total])

    pi = acos(-1.0_c_double)
    deg2rad = pi / 180.0_c_double
    rad2deg = 180.0_c_double / pi

    allocate(olon_r_v(int(ncenter)), sin_olat(int(ncenter)), cos_olat(int(ncenter)))
    allocate(sin_azim(int(nazim)), cos_azim(int(nazim)))
    allocate(sin_radi(int(nradi)), cos_radi(int(nradi)))
    allocate(dlam_base(int(nradi), int(nazim)))

!$omp parallel do private(ic) schedule(static)
    do ic = 1, int(ncenter)
        olon_r_v(ic) = olon_v(ic) * deg2rad
        sin_olat(ic) = sin(olat_v(ic) * deg2rad)
        cos_olat(ic) = cos(olat_v(ic) * deg2rad)
    enddo
!$omp end parallel do

!$omp parallel do private(ia) schedule(static)
    do ia = 1, int(nazim)
        sin_azim(ia) = sin(azim_v(ia) * deg2rad)
        cos_azim(ia) = cos(azim_v(ia) * deg2rad)
    enddo
!$omp end parallel do

!$omp parallel do private(ir) schedule(static)
    do ir = 1, int(nradi)
        sin_radi(ir) = sin(radi_v(ir) * deg2rad)
        cos_radi(ir) = cos(radi_v(ir) * deg2rad)
    enddo
!$omp end parallel do

!$omp parallel do collapse(2) private(ir, ia) schedule(static)
    do ir = 1, int(nradi)
        do ia = 1, int(nazim)
            dlam_base(ir, ia) = asin(sin_radi(ir) * sin_azim(ia))
        enddo
    enddo
!$omp end parallel do

!$omp parallel do collapse(2) private(ic, ir, ia, idx, olon_r, azim_sin, azim_cos, radi_sin, radi_cos, lats_r, dlam_r, eta_r, arg) schedule(static)
    do ic = 1, int(ncenter)
        do ir = 1, int(nradi)
            do ia = 1, int(nazim)
                idx = ((ic - 1) * int(nradi) + (ir - 1)) * int(nazim) + ia
                olon_r = olon_r_v(ic)
                azim_sin = sin_azim(ia)
                azim_cos = cos_azim(ia)
                radi_sin = sin_radi(ir)
                radi_cos = cos_radi(ir)

                lats_r = asin(sin_olat(ic) * radi_cos + cos_olat(ic) * radi_sin * azim_cos)
                dlam_r = dlam_base(ir, ia) / cos(lats_r)
                arg = sin_olat(ic) * sin(dlam_r) * azim_sin - cos(dlam_r) * azim_cos
                eta_r = acos(arg)
                if (azim_v(ia) < 180.0_c_double) then
                    eta_r = -eta_r + pi
                else
                    eta_r = eta_r + pi
                endif

                lons_v(idx) = (olon_r - dlam_r) * rad2deg
                lats_v(idx) = lats_r * rad2deg
                etas_v(idx) = eta_r
            enddo
        enddo
    enddo
!$omp end parallel do

    deallocate(olon_r_v, sin_olat, cos_olat, sin_azim, cos_azim, sin_radi, cos_radi, dlam_base)
end subroutine cylind_coords_c

subroutine project_c(n, u, v, etas, uaz, vra) bind(C, name="xvortices_project_c")
    integer(c_int), value, intent(in) :: n
    type(c_ptr), value, intent(in) :: u
    type(c_ptr), value, intent(in) :: v
    type(c_ptr), value, intent(in) :: etas
    type(c_ptr), value, intent(in) :: uaz
    type(c_ptr), value, intent(in) :: vra

    real(c_double), pointer :: u_v(:), v_v(:), etas_v(:), uaz_v(:), vra_v(:)
    integer :: i

    call c_f_pointer(u, u_v, [int(n)])
    call c_f_pointer(v, v_v, [int(n)])
    call c_f_pointer(etas, etas_v, [int(n)])
    call c_f_pointer(uaz, uaz_v, [int(n)])
    call c_f_pointer(vra, vra_v, [int(n)])

!$omp parallel do private(i) schedule(static)
    do i = 1, int(n)
        uaz_v(i) = -u_v(i) * cos(etas_v(i)) - v_v(i) * sin(etas_v(i))
        vra_v(i) = -u_v(i) * sin(etas_v(i)) + v_v(i) * cos(etas_v(i))
    enddo
!$omp end parallel do
end subroutine project_c

subroutine storm_relative_c(n, uc, vc, azim, uaz, vra, uaz_rel, vra_rel) bind(C, name="xvortices_storm_relative_c")
    integer(c_int), value, intent(in) :: n
    type(c_ptr), value, intent(in) :: uc
    type(c_ptr), value, intent(in) :: vc
    type(c_ptr), value, intent(in) :: azim
    type(c_ptr), value, intent(in) :: uaz
    type(c_ptr), value, intent(in) :: vra
    type(c_ptr), value, intent(in) :: uaz_rel
    type(c_ptr), value, intent(in) :: vra_rel

    real(c_double), pointer :: uc_v(:), vc_v(:), azim_v(:), uaz_v(:), vra_v(:)
    real(c_double), pointer :: uaz_rel_v(:), vra_rel_v(:)
    real(c_double) :: pi, deg2rad, cd, cs, ang, c_azim, c_radi
    integer :: i

    call c_f_pointer(uc, uc_v, [int(n)])
    call c_f_pointer(vc, vc_v, [int(n)])
    call c_f_pointer(azim, azim_v, [int(n)])
    call c_f_pointer(uaz, uaz_v, [int(n)])
    call c_f_pointer(vra, vra_v, [int(n)])
    call c_f_pointer(uaz_rel, uaz_rel_v, [int(n)])
    call c_f_pointer(vra_rel, vra_rel_v, [int(n)])

    pi = acos(-1.0_c_double)
    deg2rad = pi / 180.0_c_double

!$omp parallel do private(i, cd, cs, ang, c_azim, c_radi) schedule(static)
    do i = 1, int(n)
        cd = atan2(vc_v(i), uc_v(i))
        cs = hypot(uc_v(i), vc_v(i))
        ang = cd - azim_v(i) * deg2rad - pi / 2.0_c_double
        c_azim = sin(ang) * cs
        c_radi = cos(ang) * cs
        uaz_rel_v(i) = uaz_v(i) - c_azim
        vra_rel_v(i) = vra_v(i) - c_radi
    enddo
!$omp end parallel do
end subroutine storm_relative_c

subroutine interp_regular_c(nouter, nlat, nlon, ncenter, ntarget, data, lon0, dlon, lat0, dlat, center_map, target_lon, target_lat, out) bind(C, name="xvortices_interp_regular_c")
    integer(c_int), value, intent(in) :: nouter
    integer(c_int), value, intent(in) :: nlat
    integer(c_int), value, intent(in) :: nlon
    integer(c_int), value, intent(in) :: ncenter
    integer(c_int), value, intent(in) :: ntarget
    type(c_ptr), value, intent(in) :: data
    real(c_double), value, intent(in) :: lon0
    real(c_double), value, intent(in) :: dlon
    real(c_double), value, intent(in) :: lat0
    real(c_double), value, intent(in) :: dlat
    type(c_ptr), value, intent(in) :: center_map
    type(c_ptr), value, intent(in) :: target_lon
    type(c_ptr), value, intent(in) :: target_lat
    type(c_ptr), value, intent(in) :: out

    real(c_double), pointer :: data_v(:), target_lon_v(:), target_lat_v(:), out_v(:)
    integer(c_int), pointer :: center_map_v(:)
    real(c_double) :: xpos, ypos, wx, wy, v00, v01, v10, v11
    integer :: outer, target, center, ix, iy
    integer :: data_idx00, data_idx01, data_idx10, data_idx11, target_idx, out_idx

    call c_f_pointer(data, data_v, [int(nouter) * int(nlat) * int(nlon)])
    call c_f_pointer(center_map, center_map_v, [int(nouter)])
    call c_f_pointer(target_lon, target_lon_v, [int(ncenter) * int(ntarget)])
    call c_f_pointer(target_lat, target_lat_v, [int(ncenter) * int(ntarget)])
    call c_f_pointer(out, out_v, [int(nouter) * int(ntarget)])

!$omp parallel do collapse(2) private(outer, target, center, target_idx, out_idx, xpos, ypos, ix, iy, wx, wy, data_idx00, data_idx01, data_idx10, data_idx11, v00, v01, v10, v11) schedule(static)
    do outer = 1, int(nouter)
        do target = 1, int(ntarget)
            center = int(center_map_v(outer)) + 1
            target_idx = (center - 1) * int(ntarget) + target
            out_idx = (outer - 1) * int(ntarget) + target

            xpos = (target_lon_v(target_idx) - lon0) / dlon
            ypos = (target_lat_v(target_idx) - lat0) / dlat
            ix = floor(xpos) + 1
            iy = floor(ypos) + 1
            wx = xpos - floor(xpos)
            wy = ypos - floor(ypos)

            if (ix == int(nlon) .and. abs(wx) < 1.0e-12_c_double) then
                ix = int(nlon) - 1
                wx = 1.0_c_double
            endif
            if (iy == int(nlat) .and. abs(wy) < 1.0e-12_c_double) then
                iy = int(nlat) - 1
                wy = 1.0_c_double
            endif

            if (ix < 1 .or. ix >= int(nlon) .or. iy < 1 .or. iy >= int(nlat)) then
                out_v(out_idx) = ieee_value(out_v(out_idx), ieee_quiet_nan)
            else
                data_idx00 = ((outer - 1) * int(nlat) + (iy - 1)) * int(nlon) + ix
                data_idx01 = data_idx00 + 1
                data_idx10 = data_idx00 + int(nlon)
                data_idx11 = data_idx10 + 1
                v00 = data_v(data_idx00)
                v01 = data_v(data_idx01)
                v10 = data_v(data_idx10)
                v11 = data_v(data_idx11)
                out_v(out_idx) = (1.0_c_double - wx) * (1.0_c_double - wy) * v00 &
                    + wx * (1.0_c_double - wy) * v01 &
                    + (1.0_c_double - wx) * wy * v10 &
                    + wx * wy * v11
            endif
        enddo
    enddo
!$omp end parallel do
end subroutine interp_regular_c

end module xvortices_core_mod
