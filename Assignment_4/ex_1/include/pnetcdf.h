/*
 *  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 * $Id$
 *
 * src/include/pnetcdf.h.  Generated from pnetcdf.h.in by configure.
 */

#ifndef _PNETCDF_H
#define _PNETCDF_H

#include <mpi.h>

#define PNETCDF_VERSION       "1.12.0"
#define PNETCDF_VERSION_MAJOR 1
#define PNETCDF_VERSION_MINOR 12
#define PNETCDF_VERSION_SUB   0
#define PNETCDF_RELEASE_DATE  "September 30, 2019"

/* List of PnetCDF features enabled/disabled at configure time.
 * 0: disabled, 1: enabled, -1: auto
 */
#define PNETCDF_ERANGE_FILL              1
#define PNETCDF_SUBFILING                0
#define PNETCDF_RELAX_COORD_BOUND        1
#define PNETCDF_DEBUG_MODE               0
#define PNETCDF_LARGE_SINGLE_REQ         0
#define PNETCDF_NULL_BYTE_HEADER_PADDING 0
#define PNETCDF_BYTE_SWAP_IN_PLACE       -1
#define PNETCDF_BURST_BUFFERING          0
#define PNETCDF_THREAD_SAFE              0
#define PNETCDF_DRIVER_NETCDF4           0

#if defined(__cplusplus)
extern "C" {
#endif

/* NetCDF data types and their corresponding external types in C convention
 * and type names used in PnetCDF APIs:
    NC_BYTE   :   signed char       (for _schar     APIs)
    NC_CHAR   : unsigned char       (for _text      APIs)
    NC_SHORT  :   signed short int  (for _short     APIs)
    NC_INT    :   signed int        (for _int       APIs)
    NC_FLOAT  :          float      (for _float     APIs)
    NC_DOUBLE :          double     (for _double    APIs)
    NC_UBYTE  : unsigned char       (for _ubyte and
                                         _uchar     APIs)
    NC_USHORT : unsigned short int  (for _ushort    APIs)
    NC_UINT   : unsigned int        (for _uint      APIs)
    NC_INT64  :   signed long long  (for _longlong  APIs)
    NC_UINT64 : unsigned long long  (for _ulonglong APIs)
 */

/* Many constants defined in PnetCDF share the same values as NetCDF.
 * Below we keep a portion of exact same copy of netcdf.h (version 4.4.1.1)
 */
#ifndef _NETCDF_

/*! The nc_type type is just an int. */
typedef int nc_type;

/*
 *  The netcdf external data types
 */
#define	NC_NAT		0	/**< Not A Type */
#define	NC_BYTE		1	/**< signed 1 byte integer */
#define	NC_CHAR 	2	/**< ISO/ASCII character */
#define	NC_SHORT 	3	/**< signed 2 byte integer */
#define	NC_INT		4	/**< signed 4 byte integer */
#define	NC_LONG		NC_INT	/**< \deprecated required for backward compatibility. */
#define	NC_FLOAT 	5	/**< single precision floating point number */
#define	NC_DOUBLE 	6	/**< double precision floating point number */
#define	NC_UBYTE 	7	/**< unsigned 1 byte int */
#define	NC_USHORT 	8	/**< unsigned 2-byte int */
#define	NC_UINT 	9	/**< unsigned 4-byte int */
#define	NC_INT64 	10	/**< signed 8-byte int */
#define	NC_UINT64 	11	/**< unsigned 8-byte int */
#define	NC_STRING 	12	/**< string */

#define NC_MAX_ATOMIC_TYPE NC_STRING /**< @internal Largest atomic type. */

/* The following are use internally in support of user-defines
 * types. They are also the class returned by nc_inq_user_type. */
#define	NC_VLEN 	13	/**< vlen (variable-length) types */
#define	NC_OPAQUE 	14	/**< opaque types */
#define	NC_ENUM 	15	/**< enum types */
#define	NC_COMPOUND 	16	/**< compound types */

/** @internal Define the first user defined type id (leave some
 * room) */
#define NC_FIRSTUSERTYPEID 32

/** Default fill value. This is used unless _FillValue attribute
 * is set.  These values are stuffed into newly allocated space as
 * appropriate.  The hope is that one might use these to notice that a
 * particular datum has not been set. */
/**@{*/
#define NC_FILL_BYTE	((signed char)-127)
#define NC_FILL_CHAR	((char)0)
#define NC_FILL_SHORT	((short)-32767)
#define NC_FILL_INT	(-2147483647)
#define NC_FILL_FLOAT	(9.9692099683868690e+36f) /* near 15 * 2^119 */
#define NC_FILL_DOUBLE	(9.9692099683868690e+36)
#define NC_FILL_UBYTE	(255)
#define NC_FILL_USHORT	(65535)
#define NC_FILL_UINT	(4294967295U)
#define NC_FILL_INT64	((long long)-9223372036854775806LL)
#define NC_FILL_UINT64	((unsigned long long)18446744073709551614ULL)
#define NC_FILL_STRING	((char *)"")
/**@}*/

/*! Max or min values for a type. Nothing greater/smaller can be
 * stored in a netCDF file for their associated types. Recall that a C
 * compiler may define int to be any length it wants, but a NC_INT is
 * *always* a 4 byte signed int. On a platform with 64 bit ints,
 * there will be many ints which are outside the range supported by
 * NC_INT. But since NC_INT is an external format, it has to mean the
 * same thing everywhere. */
/**@{*/
#define NC_MAX_BYTE	127
#define NC_MIN_BYTE	(-NC_MAX_BYTE-1)
#define NC_MAX_CHAR	255
#define NC_MAX_SHORT	32767
#define NC_MIN_SHORT	(-NC_MAX_SHORT - 1)
#define NC_MAX_INT	2147483647
#define NC_MIN_INT	(-NC_MAX_INT - 1)
#define NC_MAX_FLOAT	3.402823466e+38f
#define NC_MIN_FLOAT	(-NC_MAX_FLOAT)
#define NC_MAX_DOUBLE	1.7976931348623157e+308
#define NC_MIN_DOUBLE	(-NC_MAX_DOUBLE)
#define NC_MAX_UBYTE	NC_MAX_CHAR
#define NC_MAX_USHORT	65535U
#define NC_MAX_UINT	4294967295U
#define NC_MAX_INT64	(9223372036854775807LL)
#define NC_MIN_INT64	(-9223372036854775807LL-1LL)
#define NC_MAX_UINT64	(18446744073709551615ULL)
/**@}*/

/** Name of fill value attribute.  If you wish a variable to use a
 * different value than the above defaults, create an attribute with
 * the same type as the variable and this reserved name. The value you
 * give the attribute will be used as the fill value for that
 * variable. */
#define _FillValue	"_FillValue"
#define NC_FILL		0	/**< Argument to nc_set_fill() to clear NC_NOFILL */
#define NC_NOFILL	0x100	/**< Argument to nc_set_fill() to turn off filling of data. */

/* Define the ioflags bits for nc_create and nc_open.
   currently unused:
        0x0002
   and the whole upper 16 bits
*/

#define NC_NOWRITE	0x0000	/**< Set read-only access for nc_open(). */
#define NC_WRITE	0x0001	/**< Set read-write access for nc_open(). */
#define NC_CLOBBER	0x0000	/**< Destroy existing file. Mode flag for nc_create(). */
#define NC_NOCLOBBER	0x0004	/**< Don't destroy existing file. Mode flag for nc_create(). */

#define NC_DISKLESS	0x0008	/**< Use diskless file. Mode flag for nc_open() or nc_create(). */
#define NC_MMAP		0x0010	/**< \deprecated Use diskless file with mmap. Mode flag for nc_open() or nc_create()*/

#define NC_64BIT_DATA	0x0020	/**< CDF-5 format: classic model but 64 bit dimensions and sizes */
#define NC_CDF5		NC_64BIT_DATA  /**< Alias NC_CDF5 to NC_64BIT_DATA */

#define NC_UDF0		0x0040	/**< User-defined format 0. */
#define NC_UDF1		0x0080	/**< User-defined format 1. */

#define NC_CLASSIC_MODEL 0x0100	/**< Enforce classic model on netCDF-4. Mode flag for nc_create(). */
#define NC_64BIT_OFFSET  0x0200	/**< Use large (64-bit) file offsets. Mode flag for nc_create(). */

/** \deprecated The following flag currently is ignored, but use in
 * nc_open() or nc_create() may someday support use of advisory
 * locking to prevent multiple writers from clobbering a file
 */
#define NC_LOCK          0x0400

/** Share updates, limit caching.
Use this in mode flags for both nc_create() and nc_open(). */
#define NC_SHARE         0x0800

#define NC_NETCDF4       0x1000  /**< Use netCDF-4/HDF5 format. Mode flag for nc_create(). */

/** The following 3 flags are deprecated as of 4.6.2. Parallel I/O is now
 * initiated by calling nc_create_par and nc_open_par, no longer by flags.
 */
#define NC_MPIIO         0x2000 /**< \deprecated */
#define NC_MPIPOSIX      NC_MPIIO /**< \deprecated */
#define NC_PNETCDF       (NC_MPIIO) /**< \deprecated */

#define NC_PERSIST       0x4000  /**< Save diskless contents to disk. Mode flag for nc_open() or nc_create() */
#define NC_INMEMORY      0x8000  /**< Read from memory. Mode flag for nc_open() or nc_create() */

#define NC_MAX_MAGIC_NUMBER_LEN 8 /**< Max len of user-defined format magic number. */

/** Format specifier for nc_set_default_format() and returned
 *  by nc_inq_format. This returns the format as provided by
 *  the API. See nc_inq_format_extended to see the true file format.
 *  Starting with version 3.6, there are different format netCDF files.
 *  4.0 introduces the third one. \see netcdf_format
 */
/**@{*/
#define NC_FORMAT_CLASSIC         (1)
/* After adding CDF5 support, the NC_FORMAT_64BIT
   flag is somewhat confusing. So, it is renamed.
   Note that the name in the contributed code
   NC_FORMAT_64BIT was renamed to NC_FORMAT_CDF2
*/
#define NC_FORMAT_64BIT_OFFSET    (2)
#define NC_FORMAT_64BIT           (NC_FORMAT_64BIT_OFFSET) /**< \deprecated Saved for compatibility.  Use NC_FORMAT_64BIT_OFFSET or NC_FORMAT_64BIT_DATA, from netCDF 4.4.0 onwards. */
#define NC_FORMAT_NETCDF4         (3)
#define NC_FORMAT_NETCDF4_CLASSIC (4)
#define NC_FORMAT_64BIT_DATA      (5)

/* Alias */
#define NC_FORMAT_CDF5    NC_FORMAT_64BIT_DATA

/* Define a mask covering format flags only */
#define NC_FORMAT_ALL (NC_64BIT_OFFSET|NC_64BIT_DATA|NC_CLASSIC_MODEL|NC_NETCDF4|NC_UDF0|NC_UDF1)

/**@}*/

/** Extended format specifier returned by  nc_inq_format_extended()
 *  Added in version 4.3.1. This returns the true format of the
 *  underlying data.
 * The function returns two values
 * 1. a small integer indicating the underlying source type
 *    of the data. Note that this may differ from what the user
 *    sees from nc_inq_format() because this latter function
 *    returns what the user can expect to see thru the API.
 * 2. A mode value indicating what mode flags are effectively
 *    set for this dataset. This usually will be a superset
 *    of the mode flags used as the argument to nc_open
 *    or nc_create.
 * More or less, the #1 values track the set of dispatch tables.
 * The #1 values are as follows.
 * Note that CDF-5 returns NC_FORMAT_NC3, but sets the mode flag properly.
 */
/**@{*/

#define NC_FORMATX_NC3       (1)
#define NC_FORMATX_NC_HDF5   (2) /**< netCDF-4 subset of HDF5 */
#define NC_FORMATX_NC4       NC_FORMATX_NC_HDF5 /**< alias */
#define NC_FORMATX_NC_HDF4   (3) /**< netCDF-4 subset of HDF4 */
#define NC_FORMATX_PNETCDF   (4)
#define NC_FORMATX_DAP2      (5)
#define NC_FORMATX_DAP4      (6)
#define NC_FORMATX_UDF0      (8)
#define NC_FORMATX_UDF1      (9)
#define NC_FORMATX_ZARR      (10)
#define NC_FORMATX_UNDEFINED (0)

  /* To avoid breaking compatibility (such as in the python library),
   we need to retain the NC_FORMAT_xxx format as well. This may come
  out eventually, as the NC_FORMATX is more clear that it's an extended
  format specifier.*/

#define NC_FORMAT_NC3       NC_FORMATX_NC3 /**< \deprecated As of 4.4.0, use NC_FORMATX_NC3 */
#define NC_FORMAT_NC_HDF5   NC_FORMATX_NC_HDF5 /**< \deprecated As of 4.4.0, use NC_FORMATX_NC_HDF5 */
#define NC_FORMAT_NC4       NC_FORMATX_NC4 /**< \deprecated As of 4.4.0, use NC_FORMATX_NC4 */
#define NC_FORMAT_NC_HDF4   NC_FORMATX_NC_HDF4 /**< \deprecated As of 4.4.0, use NC_FORMATX_HDF4 */
#define NC_FORMAT_PNETCDF   NC_FORMATX_PNETCDF /**< \deprecated As of 4.4.0, use NC_FORMATX_PNETCDF */
#define NC_FORMAT_DAP2      NC_FORMATX_DAP2 /**< \deprecated As of 4.4.0, use NC_FORMATX_DAP2 */
#define NC_FORMAT_DAP4      NC_FORMATX_DAP4 /**< \deprecated As of 4.4.0, use NC_FORMATX_DAP4 */
#define NC_FORMAT_UNDEFINED NC_FORMATX_UNDEFINED /**< \deprecated As of 4.4.0, use NC_FORMATX_UNDEFINED */

/**@}*/

/** Let nc__create() or nc__open() figure out a suitable buffer size. */
#define NC_SIZEHINT_DEFAULT 0

/** In nc__enddef(), align to the buffer size. */
#define NC_ALIGN_CHUNK ((size_t)(-1))

/** Size argument to nc_def_dim() for an unlimited dimension. */
#define NC_UNLIMITED 0L

/** Attribute id to put/get a global attribute. */
#define NC_GLOBAL -1

/**
Maximum for classic library.

In the classic netCDF model there are maximum values for the number of
dimensions in the file (\ref NC_MAX_DIMS), the number of global or per
variable attributes (\ref NC_MAX_ATTRS), the number of variables in
the file (\ref NC_MAX_VARS), and the length of a name (\ref
NC_MAX_NAME).

These maximums are enforced by the interface, to facilitate writing
applications and utilities.  However, nothing is statically allocated
to these sizes internally.

These maximums are not used for netCDF-4/HDF5 files unless they were
created with the ::NC_CLASSIC_MODEL flag.

As a rule, NC_MAX_VAR_DIMS <= NC_MAX_DIMS.

NOTE: The NC_MAX_DIMS, NC_MAX_ATTRS, and NC_MAX_VARS limits
      are *not* enforced after version 4.5.0

#define NC_MAX_DIMS     1024 (not enforced after 4.5.0)
#define NC_MAX_ATTRS    8192 (not enforced after 4.5.0)
#define NC_MAX_VARS     8192 (not enforced after 4.5.0)
#define NC_MAX_VAR_DIMS 1024 (max per variable dimensions)

*** Note the above 4 constants are different since PnetCDF 1.9.0 and NetCDF 4.5.0.

*/
/**@{*/
#define NC_MAX_DIMS	NC_MAX_INT
#define NC_MAX_ATTRS	NC_MAX_INT
#define NC_MAX_VARS	NC_MAX_INT
#define NC_MAX_NAME	256
#define NC_MAX_VAR_DIMS	NC_MAX_INT  /* max per-variable dimensions */
/**@}*/

/** This is the max size of an SD dataset name in HDF4 (from HDF4 documentation).*/
#define NC_MAX_HDF4_NAME 64

/** In HDF5 files you can set the endianness of variables with
    nc_def_var_endian(). This define is used there. */
/**@{*/
#define NC_ENDIAN_NATIVE 0
#define NC_ENDIAN_LITTLE 1
#define NC_ENDIAN_BIG    2
/**@}*/

/** In HDF5 files you can set storage for each variable to be either
 * contiguous or chunked, with nc_def_var_chunking().  This define is
 * used there. */
/**@{*/
#define NC_CHUNKED    0
#define NC_CONTIGUOUS 1
/**@}*/

/** In HDF5 files you can set check-summing for each variable.
Currently the only checksum available is Fletcher-32, which can be set
with the function nc_def_var_fletcher32.  These defines are used
there. */
/**@{*/
#define NC_NOCHECKSUM 0
#define NC_FLETCHER32 1
/**@}*/

/**@{*/
/** Control the HDF5 shuffle filter. In HDF5 files you can specify
 * that a shuffle filter should be used on each chunk of a variable to
 * improve compression for that variable. This per-variable shuffle
 * property can be set with the function nc_def_var_deflate(). */
#define NC_NOSHUFFLE 0
#define NC_SHUFFLE   1
/**@}*/

#define NC_MIN_DEFLATE_LEVEL 0 /**< Minimum deflate level. */
#define NC_MAX_DEFLATE_LEVEL 9 /**< Maximum deflate level. */

/** The netcdf version 3 functions all return integer error status.
 * These are the possible values, in addition to certain values from
 * the system errno.h.
 */
#define NC_ISSYSERR(err)	((err) > 0)

#define	NC_NOERR	0   /**< No Error */
#define	NC2_ERR         (-1)       /**< Returned for all errors in the v2 API. */

/** Not a netcdf id.

The specified netCDF ID does not refer to an
open netCDF dataset. */
#define	NC_EBADID	(-33)
#define	NC_ENFILE	(-34)	   /**< Too many netcdfs open */
#define	NC_EEXIST	(-35)	   /**< netcdf file exists && NC_NOCLOBBER */
#define	NC_EINVAL	(-36)	   /**< Invalid Argument */
#define	NC_EPERM	(-37)	   /**< Write to read only */

/** Operation not allowed in data mode. This is returned for netCDF
classic or 64-bit offset files, or for netCDF-4 files, when they were
been created with ::NC_CLASSIC_MODEL flag in nc_create(). */
#define NC_ENOTINDEFINE	(-38)

/** Operation not allowed in define mode.

The specified netCDF is in define mode rather than data mode.

With netCDF-4/HDF5 files, this error will not occur, unless
::NC_CLASSIC_MODEL was used in nc_create().
 */
#define	NC_EINDEFINE	(-39)

/** Index exceeds dimension bound.

The specified corner indices were out of range for the rank of the
specified variable. For example, a negative index or an index that is
larger than the corresponding dimension length will cause an error. */
#define	NC_EINVALCOORDS	(-40)

/** NC_MAX_DIMS exceeded. Max number of dimensions exceeded in a
classic or 64-bit offset file, or an netCDF-4 file with
::NC_CLASSIC_MODEL on. */
#define	NC_EMAXDIMS	(-41)	   /**< NC_MAX_DIMS or NC_MAX_VAR_DIMS exceeds - not enforced after 4.5.0 */

#define	NC_ENAMEINUSE	(-42)	   /**< String match to name in use */
#define NC_ENOTATT	(-43)	   /**< Attribute not found */
#define	NC_EMAXATTS	(-44)	   /**< NC_MAX_ATTRS exceeded - not enforced after 4.5.0 */
#define NC_EBADTYPE	(-45)	   /**< Not a netcdf data type */
#define NC_EBADDIM	(-46)	   /**< Invalid dimension id or name */
#define NC_EUNLIMPOS	(-47)	   /**< NC_UNLIMITED in the wrong index */

/** NC_MAX_VARS exceeded. Max number of variables exceeded in a
classic or 64-bit offset file, or an netCDF-4 file with
::NC_CLASSIC_MODEL on. */
#define	NC_EMAXVARS	(-48)	/* not enforced after 4.5.0 */

/** Variable not found.

The variable ID is invalid for the specified netCDF dataset. */
#define NC_ENOTVAR	(-49)
#define NC_EGLOBAL	(-50)	   /**< Action prohibited on NC_GLOBAL varid */
#define NC_ENOTNC	(-51)	   /**< Not a netcdf file (file format violates CDF specification) */
#define NC_ESTS        	(-52)	   /**< In Fortran, string too short */
#define NC_EMAXNAME    	(-53)	   /**< NC_MAX_NAME exceeded */
#define NC_EUNLIMIT    	(-54)	   /**< NC_UNLIMITED size already in use */
#define NC_ENORECVARS  	(-55)	   /**< nc_rec op when there are no record vars */
#define NC_ECHAR	(-56)	   /**< Attempt to convert between text & numbers */

/** Start+count exceeds dimension bound.

The specified edge lengths added to the specified corner would have
referenced data out of range for the rank of the specified
variable. For example, an edge length that is larger than the
corresponding dimension length minus the corner index will cause an
error. */
#define NC_EEDGE	(-57)      /**< Start+count exceeds dimension bound. */
#define NC_ESTRIDE	(-58)	   /**< Illegal stride */
#define NC_EBADNAME	(-59)	   /**< Attribute or variable name contains illegal characters */
/* N.B. following must match value in ncx.h */

/** Math result not representable.

One or more of the values are out of the range of values representable
by the desired type. */
#define NC_ERANGE	(-60)
#define NC_ENOMEM	(-61)	   /**< Memory allocation (malloc) failure */
#define NC_EVARSIZE     (-62)      /**< One or more variable sizes violate format constraints */
#define NC_EDIMSIZE     (-63)      /**< Invalid dimension size */
#define NC_ETRUNC       (-64)      /**< File likely truncated or possibly corrupted */
#define NC_EAXISTYPE    (-65)      /**< Unknown axis type. */

/* Following errors are added for DAP */
#define NC_EDAP         (-66)      /**< Generic DAP error */
#define NC_ECURL        (-67)      /**< Generic libcurl error */
#define NC_EIO          (-68)      /**< Generic IO error */
#define NC_ENODATA      (-69)      /**< Attempt to access variable with no data */
#define NC_EDAPSVC      (-70)      /**< DAP server error */
#define NC_EDAS		(-71)      /**< Malformed or inaccessible DAS */
#define NC_EDDS		(-72)      /**< Malformed or inaccessible DDS */
#define NC_EDMR         NC_EDDS    /**< Dap4 alias */
#define NC_EDATADDS	(-73)      /**< Malformed or inaccessible DATADDS */
#define NC_EDATADAP     NC_EDATADDS    /**< Dap4 alias */
#define NC_EDAPURL	(-74)      /**< Malformed DAP URL */
#define NC_EDAPCONSTRAINT (-75)    /**< Malformed DAP Constraint*/
#define NC_ETRANSLATION (-76)      /**< Untranslatable construct */
#define NC_EACCESS      (-77)      /**< Access Failure */
#define NC_EAUTH        (-78)      /**< Authorization Failure */

/* Misc. additional errors */
#define NC_ENOTFOUND     (-90)      /**< No such file */
#define NC_ECANTREMOVE   (-91)      /**< Can't remove file */
#define NC_EINTERNAL     (-92)      /**< NetCDF Library Internal Error */
#define NC_EPNETCDF      (-93)      /**< Error at PnetCDF layer */

/* The following was added in support of netcdf-4. Make all netcdf-4
   error codes < -100 so that errors can be added to netcdf-3 if
   needed. */
#define NC4_FIRST_ERROR  (-100)    /**< @internal All HDF5 errors < this. */
#define NC_EHDFERR       (-101)    /**< Error at HDF5 layer. */
#define NC_ECANTREAD     (-102)    /**< Can't read. */
#define NC_ECANTWRITE    (-103)    /**< Can't write. */
#define NC_ECANTCREATE   (-104)    /**< Can't create. */
#define NC_EFILEMETA     (-105)    /**< Problem with file metadata. */
#define NC_EDIMMETA      (-106)    /**< Problem with dimension metadata. */
#define NC_EATTMETA      (-107)    /**< Problem with attribute metadata. */
#define NC_EVARMETA      (-108)    /**< Problem with variable metadata. */
#define NC_ENOCOMPOUND   (-109)    /**< Not a compound type. */
#define NC_EATTEXISTS    (-110)    /**< Attribute already exists. */
#define NC_ENOTNC4       (-111)    /**< Attempting netcdf-4 operation on netcdf-3 file. */
#define NC_ESTRICTNC3    (-112)    /**< Attempting netcdf-4 operation on strict nc3 netcdf-4 file. */
#define NC_ENOTNC3       (-113)    /**< Attempting netcdf-3 operation on netcdf-4 file. */
#define NC_ENOPAR        (-114)    /**< Parallel operation on file opened for non-parallel access. */
#define NC_EPARINIT      (-115)    /**< Error initializing for parallel access. */
#define NC_EBADGRPID     (-116)    /**< Bad group ID. */
#define NC_EBADTYPID     (-117)    /**< Bad type ID. */
#define NC_ETYPDEFINED   (-118)    /**< Type has already been defined and may not be edited. */
#define NC_EBADFIELD     (-119)    /**< Bad field ID. */
#define NC_EBADCLASS     (-120)    /**< Bad class. */
#define NC_EMAPTYPE      (-121)    /**< Mapped access for atomic types only. */
#define NC_ELATEFILL     (-122)    /**< Attempt to define fill value when data already exists. */
#define NC_ELATEDEF      (-123)    /**< Attempt to define var properties, like deflate, after enddef. */
#define NC_EDIMSCALE     (-124)    /**< Problem with HDF5 dimscales. */
#define NC_ENOGRP        (-125)    /**< No group found. */
#define NC_ESTORAGE      (-126)    /**< Can't specify both contiguous and chunking. */
#define NC_EBADCHUNK     (-127)    /**< Bad chunksize. */
#define NC_ENOTBUILT     (-128)    /**< Attempt to use feature that was not turned on when netCDF was built. */
#define NC_EDISKLESS     (-129)    /**< Error in using diskless  access. */
#define NC_ECANTEXTEND   (-130)    /**< Attempt to extend dataset during ind. I/O operation. */
#define NC_EMPI          (-131)    /**< MPI operation failed. */

#define NC_EFILTER       (-132)    /**< Filter operation failed. */
#define NC_ERCFILE       (-133)    /**< RC file failure */
#define NC_ENULLPAD      (-134)    /**< Header Bytes not Null-Byte padded */
#define NC_EINMEMORY     (-135)    /**< In-memory file error */
#define NC4_LAST_ERROR   (-136)    /**< @internal All netCDF errors > this. */

/** @internal This is used in netCDF-4 files for dimensions without
 * coordinate vars. */
#define DIM_WITHOUT_VARIABLE "This is a netCDF dimension but not a netCDF variable."

/** @internal This is here at the request of the NCO team to support
 * our mistake of having chunksizes be first ints, then
 * size_t. Doh! */
#define NC_HAVE_NEW_CHUNKING_API 1

/* Errors for all remote access methods(e.g. DAP and CDMREMOTE)*/
#define NC_EURL		(NC_EDAPURL)   /**< Malformed URL */
#define NC_ECONSTRAINT  (NC_EDAPCONSTRAINT)   /**< Malformed Constraint*/

#endif
/* end of #ifndef _NETCDF_ */

/* Below are constants used in PnetCDF only */

/* invalid nonblocking request ID and zero-length request */
#define NC_REQ_NULL -1

/* indicate to flush all pending non-blocking requests */
#define NC_REQ_ALL     -1
#define NC_GET_REQ_ALL -2
#define NC_PUT_REQ_ALL -3

/* max number of opened files allowed */
#define NC_MAX_NFILES	1024

#define NC_FORMAT_UNKNOWN -1

/* CDF version 1, NC_32BIT is used internally and never
   actually passed in to ncmpi_create  */
#define NC_32BIT	0x1000000

/* CDF-5 format, (64-bit) supported */
#ifndef NC_64BIT_DATA
#define NC_64BIT_DATA	0x0020
#endif

/* CDF-2 format, with NC_64BIT_OFFSET. */
#ifndef NC_FORMAT_CDF2
#define NC_FORMAT_CDF2  2
#endif

/* CDF-5 format, with NC_64BIT_DATA. */
#ifndef NC_FORMAT_CDF5
#define NC_FORMAT_CDF5  5
#endif

#define NC_BP        0x10000  /**< Use ADIOS BP format. */
#define NC_FORMAT_BP 6

#ifndef NC_ENULLPAD
#define NC_ENULLPAD      (-134)    /**< Header Bytes not Null-Byte padded */
#endif

/* PnetCDF Error Codes: */
#define NC_ESMALL			(-201) /**< size of MPI_Offset too small for format */
#define NC_ENOTINDEP			(-202) /**< Operation not allowed in collective data mode */
#define NC_EINDEP			(-203) /**< Operation not allowed in independent data mode */
#define NC_EFILE			(-204) /**< Unknown error in file operation */
#define NC_EREAD			(-205) /**< Unknown error in reading file */
#define NC_EWRITE			(-206) /**< Unknown error in writing to file */
#define NC_EOFILE			(-207) /**< file open/creation failed */
#define NC_EMULTITYPES			(-208) /**< Multiple etypes used in MPI datatype */
#define NC_EIOMISMATCH			(-209) /**< Input/Output data amount mismatch */
#define NC_ENEGATIVECNT			(-210) /**< Negative count is specified */
#define NC_EUNSPTETYPE			(-211) /**< Unsupported etype in memory MPI datatype */
#define NC_EINVAL_REQUEST		(-212) /**< invalid nonblocking request ID */
#define NC_EAINT_TOO_SMALL		(-213) /**< MPI_Aint not large enough to hold requested value */
#define NC_ENOTSUPPORT			(-214) /**< feature is not yet supported */
#define NC_ENULLBUF			(-215) /**< trying to attach a NULL buffer */
#define NC_EPREVATTACHBUF		(-216) /**< previous attached buffer is found */
#define NC_ENULLABUF			(-217) /**< no attached buffer is found */
#define NC_EPENDINGBPUT			(-218) /**< pending bput is found, cannot detach buffer */
#define NC_EINSUFFBUF			(-219) /**< attached buffer is too small */
#define NC_ENOENT			(-220) /**< File does not exist */
#define NC_EINTOVERFLOW			(-221) /**< Overflow when type cast to 4-byte integer */
#define NC_ENOTENABLED			(-222) /**< feature is not enabled */
#define NC_EBAD_FILE			(-223) /**< Invalid file name (e.g., path name too long) */
#define NC_ENO_SPACE			(-224) /**< Not enough space */
#define NC_EQUOTA			(-225) /**< Quota exceeded */
#define NC_ENULLSTART			(-226) /**< argument start is a NULL pointer */
#define NC_ENULLCOUNT			(-227) /**< argument count is a NULL pointer */
#define NC_EINVAL_CMODE			(-228) /**< Invalid file create mode */
#define NC_ETYPESIZE			(-229) /**< MPI derived data type size error (bigger than the variable size) */
#define NC_ETYPE_MISMATCH		(-230) /**< element type of the MPI derived data type mismatches the variable type */
#define NC_ETYPESIZE_MISMATCH		(-231) /**< file type size mismatches buffer type size */
#define NC_ESTRICTCDF2			(-232) /**< Attempting CDF-5 operation on CDF-2 file */
#define NC_ENOTRECVAR			(-233) /**< Attempting operation only for record variables */
#define NC_ENOTFILL			(-234) /**< Attempting to fill a variable when its fill mode is off */
#define NC_EINVAL_OMODE			(-235) /**< Invalid file open mode */
#define NC_EPENDING			(-236) /**< Pending nonblocking request is found at file close */
#define NC_EMAX_REQ			(-237) /**< Size of I/O request exceeds INT_MAX */
#define NC_EBADLOG			(-238) /**< Unrecognized log file format */
#define NC_EFLUSHED			(-239) /**< Nonblocking request has already been flushed to the PFS. It is too late to cancel */
#define NC_EADIOS		      (-240) /**< unknown ADIOS error */ 
/* add new error here */

/* header inconsistency errors start from -250 */
#define NC_EMULTIDEFINE			(-250) /**< NC definitions inconsistent among processes */
#define NC_EMULTIDEFINE_OMODE		(-251) /**< inconsistent file open modes among processes */
#define NC_EMULTIDEFINE_DIM_NUM		(-252) /**< inconsistent number of dimensions */
#define NC_EMULTIDEFINE_DIM_SIZE	(-253) /**< inconsistent size of dimension */
#define NC_EMULTIDEFINE_DIM_NAME	(-254) /**< inconsistent dimension names */
#define NC_EMULTIDEFINE_VAR_NUM		(-255) /**< inconsistent number of variables */
#define NC_EMULTIDEFINE_VAR_NAME	(-256) /**< inconsistent variable name */
#define NC_EMULTIDEFINE_VAR_NDIMS	(-257) /**< inconsistent variable's number of dimensions */
#define NC_EMULTIDEFINE_VAR_DIMIDS	(-258) /**< inconsistent variable's dimension IDs */
#define NC_EMULTIDEFINE_VAR_TYPE	(-259) /**< inconsistent variable's data type */
#define NC_EMULTIDEFINE_VAR_LEN		(-260) /**< inconsistent variable's size */
#define NC_EMULTIDEFINE_NUMRECS		(-261) /**< inconsistent number of records */
#define NC_EMULTIDEFINE_VAR_BEGIN	(-262) /**< inconsistent variable file begin offset (internal use) */
#define NC_EMULTIDEFINE_ATTR_NUM	(-263) /**< inconsistent number of attributes */
#define NC_EMULTIDEFINE_ATTR_SIZE	(-264) /**< inconsistent memory space used by attribute (internal use) */
#define NC_EMULTIDEFINE_ATTR_NAME	(-265) /**< inconsistent attribute name */
#define NC_EMULTIDEFINE_ATTR_TYPE	(-266) /**< inconsistent attribute type */
#define NC_EMULTIDEFINE_ATTR_LEN	(-267) /**< inconsistent attribute length */
#define NC_EMULTIDEFINE_ATTR_VAL	(-268) /**< inconsistent attribute value */
#define NC_EMULTIDEFINE_FNC_ARGS	(-269) /**< inconsistent function arguments used in collective API */
#define NC_EMULTIDEFINE_FILL_MODE	(-270) /**< inconsistent dataset fill mode */
#define NC_EMULTIDEFINE_VAR_FILL_MODE	(-271) /**< inconsistent variable fill mode */
#define NC_EMULTIDEFINE_VAR_FILL_VALUE	(-272) /**< inconsistent variable fill value */
#define NC_EMULTIDEFINE_CMODE		(-273) /**< inconsistent file create modes among processes */

#define NC_EMULTIDEFINE_FIRST		NC_EMULTIDEFINE
#define NC_EMULTIDEFINE_LAST		NC_EMULTIDEFINE_CMODE

/* backward compatible with PnetCDF 1.3.1 and earlier */
#define NC_ECMODE			NC_EMULTIDEFINE_OMODE
#define NC_EDIMS_NELEMS_MULTIDEFINE	NC_EMULTIDEFINE_DIM_NUM
#define NC_EDIMS_SIZE_MULTIDEFINE	NC_EMULTIDEFINE_DIM_SIZE
#define NC_EDIMS_NAME_MULTIDEFINE	NC_EMULTIDEFINE_DIM_NAME
#define NC_EVARS_NELEMS_MULTIDEFINE	NC_EMULTIDEFINE_VAR_NUM
#define NC_EVARS_NAME_MULTIDEFINE	NC_EMULTIDEFINE_VAR_NAME
#define NC_EVARS_NDIMS_MULTIDEFINE	NC_EMULTIDEFINE_VAR_NDIMS
#define NC_EVARS_DIMIDS_MULTIDEFINE	NC_EMULTIDEFINE_VAR_DIMIDS
#define NC_EVARS_TYPE_MULTIDEFINE	NC_EMULTIDEFINE_VAR_TYPE
#define NC_EVARS_LEN_MULTIDEFINE	NC_EMULTIDEFINE_VAR_LEN
#define NC_ENUMRECS_MULTIDEFINE		NC_EMULTIDEFINE_NUMRECS
#define NC_EVARS_BEGIN_MULTIDEFINE	NC_EMULTIDEFINE_VAR_BEGIN


/*
 * The Interface
 */

/* Begin Prototypes */

extern const char*
ncmpi_strerror(int err);

extern const char*
ncmpi_strerrno(int err);

/* Begin File Functions */

extern int
ncmpi_create(MPI_Comm comm, const char *path, int cmode, MPI_Info info,
             int *ncidp);

extern int
ncmpi_open(MPI_Comm comm, const char *path, int omode, MPI_Info info,
           int *ncidp);

extern int
ncmpi_inq_file_info(int ncid, MPI_Info *info_used);

extern int
ncmpi_get_file_info(int ncid, MPI_Info *info_used); /* deprecated */

extern int
ncmpi_delete(const char *filename, MPI_Info info);

extern int
ncmpi_enddef(int ncid);

extern int
ncmpi__enddef(int ncid, MPI_Offset h_minfree, MPI_Offset v_align,
              MPI_Offset v_minfree, MPI_Offset r_align);

extern int
ncmpi_redef(int ncid);

extern int
ncmpi_set_default_format(int format, int *old_formatp);

extern int
ncmpi_inq_default_format(int *formatp);

extern int
ncmpi_sync(int ncid);

extern int
ncmpi_flush(int ncid);

extern int
ncmpi_sync_numrecs(int ncid);

extern int
ncmpi_abort(int ncid);

extern int
ncmpi_begin_indep_data(int ncid);

extern int
ncmpi_end_indep_data(int ncid);

extern int
ncmpi_close(int ncid);

extern int
ncmpi_set_fill(int ncid, int fillmode, int *old_modep);

extern int
ncmpi_def_var_fill(int ncid, int varid, int no_fill, const void *fill_value);

extern int
ncmpi_fill_var_rec(int ncid, int varid, MPI_Offset recno);

/* End File Functions */

/* Begin Define Mode Functions */

extern int
ncmpi_def_dim(int ncid, const char *name, MPI_Offset len, int *idp);

extern int
ncmpi_def_var(int ncid, const char *name, nc_type xtype, int ndims,
              const int *dimidsp, int *varidp);

extern int
ncmpi_rename_dim(int ncid, int dimid, const char *name);

extern int
ncmpi_rename_var(int ncid, int varid, const char *name);

/* End Define Mode Functions */

/* Begin Inquiry Functions */

const char* ncmpi_inq_libvers(void);

extern int
ncmpi_inq(int ncid, int *ndimsp, int *nvarsp, int *ngattsp, int *unlimdimidp);

extern int
ncmpi_inq_format(int ncid, int *formatp);

extern int
ncmpi_inq_file_format(const char *filename, int *formatp);

extern int
ncmpi_inq_version(int ncid, int *NC_mode);

extern int
ncmpi_inq_striping(int ncid, int *striping_size, int *striping_count);

extern int
ncmpi_inq_ndims(int ncid, int *ndimsp);

extern int
ncmpi_inq_nvars(int ncid, int *nvarsp);

extern int
ncmpi_inq_num_rec_vars(int ncid, int *nvarsp);

extern int
ncmpi_inq_num_fix_vars(int ncid, int *nvarsp);

extern int
ncmpi_inq_natts(int ncid, int *ngattsp);

extern int
ncmpi_inq_unlimdim(int ncid, int *unlimdimidp);

extern int
ncmpi_inq_dimid(int ncid, const char *name, int *idp);

extern int
ncmpi_inq_dim(int ncid, int dimid, char *name, MPI_Offset *lenp);

extern int
ncmpi_inq_dimname(int ncid, int dimid, char *name);

extern int
ncmpi_inq_dimlen(int ncid, int dimid, MPI_Offset *lenp);

extern int
ncmpi_inq_var(int ncid, int varid, char *name, nc_type *xtypep, int *ndimsp,
              int *dimidsp, int *nattsp);

extern int
ncmpi_inq_varid(int ncid, const char *name, int *varidp);

extern int
ncmpi_inq_varname(int ncid, int varid, char *name);

extern int
ncmpi_inq_vartype(int ncid, int varid, nc_type *xtypep);

extern int
ncmpi_inq_varndims(int ncid, int varid, int *ndimsp);

extern int
ncmpi_inq_vardimid(int ncid, int varid, int *dimidsp);

extern int
ncmpi_inq_varnatts(int ncid, int varid, int *nattsp);

extern int
ncmpi_inq_varoffset(int ncid, int varid, MPI_Offset *offset);

extern int
ncmpi_inq_put_size(int ncid, MPI_Offset *size);

extern int
ncmpi_inq_get_size(int ncid, MPI_Offset *size);

extern int
ncmpi_inq_header_size(int ncid, MPI_Offset *size);

extern int
ncmpi_inq_header_extent(int ncid, MPI_Offset *extent);

extern int
ncmpi_inq_malloc_size(MPI_Offset *size);

extern int
ncmpi_inq_malloc_max_size(MPI_Offset *size);

extern int
ncmpi_inq_malloc_list(void);

extern int
ncmpi_inq_files_opened(int *num, int *ncids);

extern int
ncmpi_inq_recsize(int ncid, MPI_Offset *recsize);

extern int
ncmpi_inq_var_fill(int ncid, int varid, int *no_fill, void *fill_value);

extern int
ncmpi_inq_path(int ncid, int *pathlen, char *path);

/* End Inquiry Functions */

/* Begin _att */

extern int
ncmpi_inq_att(int ncid, int varid, const char *name, nc_type *xtypep,
              MPI_Offset *lenp);

extern int
ncmpi_inq_attid(int ncid, int varid, const char *name, int *idp);

extern int
ncmpi_inq_atttype(int ncid, int varid, const char *name, nc_type *xtypep);

extern int
ncmpi_inq_attlen(int ncid, int varid, const char *name, MPI_Offset *lenp);

extern int
ncmpi_inq_attname(int ncid, int varid, int attnum, char *name);

extern int
ncmpi_copy_att(int ncid_in, int varid_in, const char *name, int ncid_out,
               int varid_out);

extern int
ncmpi_rename_att(int ncid, int varid, const char *name, const char *newname);

extern int
ncmpi_del_att(int ncid, int varid, const char *name);

extern int
ncmpi_put_att(int ncid, int varid, const char *name, nc_type xtype,
              MPI_Offset nelems, const void *value);

extern int
ncmpi_put_att_text(int ncid, int varid, const char *name, MPI_Offset len,
              const char *op);

extern int
ncmpi_put_att_schar(int ncid, int varid, const char *name, nc_type xtype,
              MPI_Offset len, const signed char *op);

extern int
ncmpi_put_att_short(int ncid, int varid, const char *name, nc_type xtype,
              MPI_Offset len, const short *op);

extern int
ncmpi_put_att_int(int ncid, int varid, const char *name, nc_type xtype,
              MPI_Offset len, const int *op);

extern int
ncmpi_put_att_float(int ncid, int varid, const char *name, nc_type xtype,
              MPI_Offset len, const float *op);

extern int
ncmpi_put_att_double(int ncid, int varid, const char *name, nc_type xtype,
              MPI_Offset len, const double *op);

extern int
ncmpi_put_att_longlong(int ncid, int varid, const char *name, nc_type xtype,
              MPI_Offset len, const long long *op);

extern int
ncmpi_get_att(int ncid, int varid, const char *name, void *value);

extern int
ncmpi_get_att_text(int ncid, int varid, const char *name, char *ip);

extern int
ncmpi_get_att_schar(int ncid, int varid, const char *name, signed char *ip);

extern int
ncmpi_get_att_short(int ncid, int varid, const char *name, short *ip);

extern int
ncmpi_get_att_int(int ncid, int varid, const char *name, int *ip);

extern int
ncmpi_get_att_float(int ncid, int varid, const char *name, float *ip);

extern int
ncmpi_get_att_double(int ncid, int varid, const char *name, double *ip);

extern int
ncmpi_get_att_longlong(int ncid, int varid, const char *name, long long *ip);

/* Begin Skip Prototypes for Fortran binding */
/* skip types: uchar, ubyte, ushort, uint, long, ulonglong string */

extern int
ncmpi_put_att_uchar(int ncid, int varid, const char *name, nc_type xtype,
              MPI_Offset len, const unsigned char *op);

extern int
ncmpi_put_att_ubyte(int ncid, int varid, const char *name, nc_type xtype,
              MPI_Offset len, const unsigned char *op);

extern int
ncmpi_put_att_ushort(int ncid, int varid, const char *name, nc_type xtype,
              MPI_Offset len, const unsigned short *op);

extern int
ncmpi_put_att_uint(int ncid, int varid, const char *name, nc_type xtype,
              MPI_Offset len, const unsigned int *op);

extern int
ncmpi_put_att_long(int ncid, int varid, const char *name, nc_type xtype,
              MPI_Offset len, const long *op);

extern int
ncmpi_put_att_ulonglong(int ncid, int varid, const char *name, nc_type xtype,
              MPI_Offset len, const unsigned long long *op);

extern int
ncmpi_get_att_uchar(int ncid, int varid, const char *name, unsigned char *ip);

extern int
ncmpi_get_att_ubyte(int ncid, int varid, const char *name, unsigned char *ip);

extern int
ncmpi_get_att_ushort(int ncid, int varid, const char *name, unsigned short *ip);

extern int
ncmpi_get_att_uint(int ncid, int varid, const char *name, unsigned int *ip);

extern int
ncmpi_get_att_long(int ncid, int varid, const char *name, long *ip);

extern int
ncmpi_get_att_ulonglong(int ncid, int varid, const char *name,
              unsigned long long *ip);

/* End Skip Prototypes for Fortran binding */

/* End _att */

/* Begin {put,get}_var1 */

extern int
ncmpi_put_var1(int ncid, int varid, const MPI_Offset *start,
               const void *op, MPI_Offset bufcount, MPI_Datatype buftype);
extern int
ncmpi_put_var1_all(int ncid, int varid, const MPI_Offset *start,
               const void *op, MPI_Offset bufcount, MPI_Datatype buftype);

extern int
ncmpi_put_var1_text(int ncid, int varid, const MPI_Offset *start,
               const char *op);
extern int
ncmpi_put_var1_text_all(int ncid, int varid, const MPI_Offset *start,
               const char *op);

extern int
ncmpi_put_var1_schar(int ncid, int varid, const MPI_Offset *start,
               const signed char *op);
extern int
ncmpi_put_var1_schar_all(int ncid, int varid, const MPI_Offset *start,
               const signed char *op);

extern int
ncmpi_put_var1_short(int ncid, int varid, const MPI_Offset *start,
               const short *op);
extern int
ncmpi_put_var1_short_all(int ncid, int varid, const MPI_Offset *start,
               const short *op);

extern int
ncmpi_put_var1_int(int ncid, int varid, const MPI_Offset *start,
               const int *op);
extern int
ncmpi_put_var1_int_all(int ncid, int varid, const MPI_Offset *start,
               const int *op);

extern int
ncmpi_put_var1_float(int ncid, int varid, const MPI_Offset *start,
               const float *op);
extern int
ncmpi_put_var1_float_all(int ncid, int varid, const MPI_Offset *start,
               const float *op);

extern int
ncmpi_put_var1_double(int ncid, int varid, const MPI_Offset *start,
               const double *op);
extern int
ncmpi_put_var1_double_all(int ncid, int varid, const MPI_Offset *start,
               const double *op);

extern int
ncmpi_put_var1_longlong(int ncid, int varid, const MPI_Offset *start,
               const long long *op);
extern int
ncmpi_put_var1_longlong_all(int ncid, int varid, const MPI_Offset *start,
               const long long *op);

extern int
ncmpi_get_var1(int ncid, int varid, const MPI_Offset *start,
               void *ip, MPI_Offset bufcount, MPI_Datatype buftype);
extern int
ncmpi_get_var1_all(int ncid, int varid, const MPI_Offset *start,
               void *ip, MPI_Offset bufcount, MPI_Datatype buftype);

extern int
ncmpi_get_var1_text(int ncid, int varid, const MPI_Offset *start,
               char *ip);
extern int
ncmpi_get_var1_text_all(int ncid, int varid, const MPI_Offset *start,
               char *ip);

extern int
ncmpi_get_var1_schar(int ncid, int varid, const MPI_Offset *start,
               signed char *ip);
extern int
ncmpi_get_var1_schar_all(int ncid, int varid, const MPI_Offset *start,
               signed char *ip);

extern int
ncmpi_get_var1_short(int ncid, int varid, const MPI_Offset *start,
               short *ip);
extern int
ncmpi_get_var1_short_all(int ncid, int varid, const MPI_Offset *start,
               short *ip);

extern int
ncmpi_get_var1_int(int ncid, int varid, const MPI_Offset *start,
               int *ip);
extern int
ncmpi_get_var1_int_all(int ncid, int varid, const MPI_Offset *start,
               int *ip);

extern int
ncmpi_get_var1_float(int ncid, int varid, const MPI_Offset *start,
               float *ip);
extern int
ncmpi_get_var1_float_all(int ncid, int varid, const MPI_Offset *start,
               float *ip);

extern int
ncmpi_get_var1_double(int ncid, int varid, const MPI_Offset *start,
               double *ip);
extern int
ncmpi_get_var1_double_all(int ncid, int varid, const MPI_Offset *start,
               double *ip);

extern int
ncmpi_get_var1_longlong(int ncid, int varid, const MPI_Offset *start,
               long long *ip);
extern int
ncmpi_get_var1_longlong_all(int ncid, int varid, const MPI_Offset *start,
               long long *ip);

/* Begin Skip Prototypes for Fortran binding */
/* skip types: uchar, ubyte, ushort, uint, long, ulonglong string */

extern int
ncmpi_put_var1_uchar(int ncid, int varid, const MPI_Offset *start,
               const unsigned char *op);
extern int
ncmpi_put_var1_uchar_all(int ncid, int varid, const MPI_Offset *start,
               const unsigned char *op);

extern int
ncmpi_put_var1_ushort(int ncid, int varid, const MPI_Offset *start,
               const unsigned short *op);
extern int
ncmpi_put_var1_ushort_all(int ncid, int varid, const MPI_Offset *start,
               const unsigned short *op);

extern int
ncmpi_put_var1_uint(int ncid, int varid, const MPI_Offset *start,
               const unsigned int *op);
extern int
ncmpi_put_var1_uint_all(int ncid, int varid, const MPI_Offset *start,
               const unsigned int *op);

extern int
ncmpi_put_var1_long(int ncid, int varid, const MPI_Offset *start,
               const long *ip);
extern int
ncmpi_put_var1_long_all(int ncid, int varid, const MPI_Offset *start,
               const long *ip);

extern int
ncmpi_put_var1_ulonglong(int ncid, int varid, const MPI_Offset *start,
               const unsigned long long *ip);
extern int
ncmpi_put_var1_ulonglong_all(int ncid, int varid, const MPI_Offset *start,
               const unsigned long long *ip);

extern int
ncmpi_get_var1_uchar(int ncid, int varid, const MPI_Offset *start,
               unsigned char *ip);
extern int
ncmpi_get_var1_uchar_all(int ncid, int varid, const MPI_Offset *start,
               unsigned char *ip);

extern int
ncmpi_get_var1_ushort(int ncid, int varid, const MPI_Offset *start,
               unsigned short *ip);
extern int
ncmpi_get_var1_ushort_all(int ncid, int varid, const MPI_Offset *start,
               unsigned short *ip);

extern int
ncmpi_get_var1_uint(int ncid, int varid, const MPI_Offset *start,
               unsigned int *ip);
extern int
ncmpi_get_var1_uint_all(int ncid, int varid, const MPI_Offset *start,
               unsigned int *ip);

extern int
ncmpi_get_var1_long(int ncid, int varid, const MPI_Offset *start,
               long *ip);
extern int
ncmpi_get_var1_long_all(int ncid, int varid, const MPI_Offset *start,
               long *ip);

extern int
ncmpi_get_var1_ulonglong(int ncid, int varid, const MPI_Offset *start,
               unsigned long long *ip);
extern int
ncmpi_get_var1_ulonglong_all(int ncid, int varid, const MPI_Offset *start,
               unsigned long long *ip);
/* End Skip Prototypes for Fortran binding */

/* End {put,get}_var1 */

/* Begin {put,get}_var */

extern int
ncmpi_put_var(int ncid, int varid, const void *op, MPI_Offset bufcount,
              MPI_Datatype buftype);

extern int
ncmpi_put_var_all(int ncid, int varid, const void *op, MPI_Offset bufcount,
              MPI_Datatype buftype);

extern int
ncmpi_put_var_text(int ncid, int varid, const char *op);
extern int
ncmpi_put_var_text_all(int ncid, int varid, const char *op);

extern int
ncmpi_put_var_schar(int ncid, int varid, const signed char *op);
extern int
ncmpi_put_var_schar_all(int ncid, int varid, const signed char *op);

extern int
ncmpi_put_var_short(int ncid, int varid, const short *op);
extern int
ncmpi_put_var_short_all(int ncid, int varid, const short *op);

extern int
ncmpi_put_var_int(int ncid, int varid, const int *op);
extern int
ncmpi_put_var_int_all(int ncid, int varid, const int *op);

extern int
ncmpi_put_var_float(int ncid, int varid, const float *op);
extern int
ncmpi_put_var_float_all(int ncid, int varid, const float *op);

extern int
ncmpi_put_var_double(int ncid, int varid, const double *op);
extern int
ncmpi_put_var_double_all(int ncid, int varid, const double *op);

extern int
ncmpi_put_var_longlong(int ncid, int varid, const long long *op);
extern int
ncmpi_put_var_longlong_all(int ncid, int varid, const long long *op);

extern int
ncmpi_get_var(int ncid, int varid, void *ip, MPI_Offset bufcount,
              MPI_Datatype buftype);
extern int
ncmpi_get_var_all(int ncid, int varid, void *ip, MPI_Offset bufcount,
              MPI_Datatype buftype);

extern int
ncmpi_get_var_text(int ncid, int varid, char *ip);
extern int
ncmpi_get_var_text_all(int ncid, int varid, char *ip);

extern int
ncmpi_get_var_schar(int ncid, int varid, signed char *ip);
extern int
ncmpi_get_var_schar_all(int ncid, int varid, signed char *ip);

extern int
ncmpi_get_var_short(int ncid, int varid, short *ip);
extern int
ncmpi_get_var_short_all(int ncid, int varid, short *ip);

extern int
ncmpi_get_var_int(int ncid, int varid, int *ip);
extern int
ncmpi_get_var_int_all(int ncid, int varid, int *ip);

extern int
ncmpi_get_var_float(int ncid, int varid, float *ip);
extern int
ncmpi_get_var_float_all(int ncid, int varid, float *ip);

extern int
ncmpi_get_var_double(int ncid, int varid, double *ip);
extern int
ncmpi_get_var_double_all(int ncid, int varid, double *ip);

extern int
ncmpi_get_var_longlong(int ncid, int varid, long long *ip);
extern int
ncmpi_get_var_longlong_all(int ncid, int varid, long long *ip);

/* Begin Skip Prototypes for Fortran binding */
/* skip types: uchar, ubyte, ushort, uint, long, ulonglong string */

extern int
ncmpi_put_var_uchar(int ncid, int varid, const unsigned char *op);
extern int
ncmpi_put_var_uchar_all(int ncid, int varid, const unsigned char *op);

extern int
ncmpi_put_var_ushort(int ncid, int varid, const unsigned short *op);
extern int
ncmpi_put_var_ushort_all(int ncid, int varid, const unsigned short *op);

extern int
ncmpi_put_var_uint(int ncid, int varid, const unsigned int *op);
extern int
ncmpi_put_var_uint_all(int ncid, int varid, const unsigned int *op);

extern int
ncmpi_put_var_long(int ncid, int varid, const long *op);
extern int
ncmpi_put_var_long_all(int ncid, int varid, const long *op);

extern int
ncmpi_put_var_ulonglong(int ncid, int varid, const unsigned long long *op);
extern int
ncmpi_put_var_ulonglong_all(int ncid, int varid, const unsigned long long *op);

extern int
ncmpi_get_var_uchar(int ncid, int varid, unsigned char *ip);
extern int
ncmpi_get_var_uchar_all(int ncid, int varid, unsigned char *ip);

extern int
ncmpi_get_var_ushort(int ncid, int varid, unsigned short *ip);
extern int
ncmpi_get_var_ushort_all(int ncid, int varid, unsigned short *ip);

extern int
ncmpi_get_var_uint(int ncid, int varid, unsigned int *ip);
extern int
ncmpi_get_var_uint_all(int ncid, int varid, unsigned int *ip);

extern int
ncmpi_get_var_long(int ncid, int varid, long *ip);
extern int
ncmpi_get_var_long_all(int ncid, int varid, long *ip);

extern int
ncmpi_get_var_ulonglong(int ncid, int varid, unsigned long long *ip);
extern int
ncmpi_get_var_ulonglong_all(int ncid, int varid, unsigned long long *ip);
/* End Skip Prototypes for Fortran binding */

/* End {put,get}_var */

/* Begin {put,get}_vara */

extern int
ncmpi_put_vara(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const void *op,
               MPI_Offset bufcount, MPI_Datatype buftype);
extern int
ncmpi_put_vara_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const void *op,
               MPI_Offset bufcount, MPI_Datatype buftype);

extern int
ncmpi_put_vara_text(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const char *op);
extern int
ncmpi_put_vara_text_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const char *op);

extern int
ncmpi_put_vara_schar(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const signed char *op);
extern int
ncmpi_put_vara_schar_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const signed char *op);

extern int
ncmpi_put_vara_short(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const short *op);
extern int
ncmpi_put_vara_short_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const short *op);

extern int
ncmpi_put_vara_int(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const int *op);
extern int
ncmpi_put_vara_int_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const int *op);

extern int
ncmpi_put_vara_float(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const float *op);

extern int
ncmpi_put_vara_float_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const float *op);

extern int
ncmpi_put_vara_double(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const double *op);
extern int
ncmpi_put_vara_double_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const double *op);

extern int
ncmpi_put_vara_longlong(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const long long *op);
extern int
ncmpi_put_vara_longlong_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const long long *op);

extern int
ncmpi_get_vara(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, void *ip, MPI_Offset bufcount,
               MPI_Datatype buftype);
extern int
ncmpi_get_vara_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, void *ip, MPI_Offset bufcount,
               MPI_Datatype buftype);

extern int
ncmpi_get_vara_text(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, char *ip);
extern int
ncmpi_get_vara_text_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, char *ip);

extern int
ncmpi_get_vara_schar(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, signed char *ip);
extern int
ncmpi_get_vara_schar_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, signed char *ip);

extern int
ncmpi_get_vara_short(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, short *ip);
extern int
ncmpi_get_vara_short_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, short *ip);

extern int
ncmpi_get_vara_int(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, int *ip);
extern int
ncmpi_get_vara_int_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, int *ip);

extern int
ncmpi_get_vara_float(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, float *ip);
extern int
ncmpi_get_vara_float_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, float *ip);

extern int
ncmpi_get_vara_double(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, double *ip);
extern int
ncmpi_get_vara_double_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, double *ip);

extern int
ncmpi_get_vara_longlong(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, long long *ip);
extern int
ncmpi_get_vara_longlong_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, long long *ip);

/* Begin Skip Prototypes for Fortran binding */
/* skip types: uchar, ubyte, ushort, uint, long, ulonglong string */

extern int
ncmpi_put_vara_uchar(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const unsigned char *op);
extern int
ncmpi_put_vara_uchar_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const unsigned char *op);

extern int
ncmpi_put_vara_ushort(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const unsigned short *op);
extern int
ncmpi_put_vara_ushort_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const unsigned short *op);

extern int
ncmpi_put_vara_uint(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const unsigned int *op);
extern int
ncmpi_put_vara_uint_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const unsigned int *op);

extern int
ncmpi_put_vara_long(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const long *op);
extern int
ncmpi_put_vara_long_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const long *op);

extern int
ncmpi_put_vara_ulonglong(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const unsigned long long *op);
extern int
ncmpi_put_vara_ulonglong_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const unsigned long long *op);

extern int
ncmpi_get_vara_uchar(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, unsigned char *ip);
extern int
ncmpi_get_vara_uchar_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, unsigned char *ip);

extern int
ncmpi_get_vara_ushort(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, unsigned short *ip);
extern int
ncmpi_get_vara_ushort_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, unsigned short *ip);

extern int
ncmpi_get_vara_uint(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, unsigned int *ip);
extern int
ncmpi_get_vara_uint_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, unsigned int *ip);

extern int
ncmpi_get_vara_long(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, long *ip);
extern int
ncmpi_get_vara_long_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, long *ip);

extern int
ncmpi_get_vara_ulonglong(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, unsigned long long *ip);
extern int
ncmpi_get_vara_ulonglong_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, unsigned long long *ip);

/* End Skip Prototypes for Fortran binding */

/* End {put,get}_vara */

/* Begin {put,get}_vars */

extern int
ncmpi_put_vars(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const void *op, MPI_Offset bufcount, MPI_Datatype buftype);
extern int
ncmpi_put_vars_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const void *op, MPI_Offset bufcount, MPI_Datatype buftype);

extern int
ncmpi_put_vars_text(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const char *op);
extern int
ncmpi_put_vars_text_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const char *op);

extern int
ncmpi_put_vars_schar(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const signed char *op);
extern int
ncmpi_put_vars_schar_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const signed char *op);

extern int
ncmpi_put_vars_short(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const short *op);
extern int
ncmpi_put_vars_short_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const short *op);

extern int
ncmpi_put_vars_int(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const int *op);
extern int
ncmpi_put_vars_int_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const int *op);

extern int
ncmpi_put_vars_float(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const float *op);
extern int
ncmpi_put_vars_float_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const float *op);

extern int
ncmpi_put_vars_double(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const double *op);
extern int
ncmpi_put_vars_double_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const double *op);

extern int
ncmpi_put_vars_longlong(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const long long *op);
extern int
ncmpi_put_vars_longlong_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const long long *op);

extern int
ncmpi_get_vars(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               void *ip, MPI_Offset bufcount, MPI_Datatype buftype);
extern int
ncmpi_get_vars_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               void *ip, MPI_Offset bufcount, MPI_Datatype buftype);

extern int
ncmpi_get_vars_schar(int ncid, int varid, const MPI_Offset *start,
                   const MPI_Offset *count, const MPI_Offset *stride,
                   signed char *ip);
extern int
ncmpi_get_vars_schar_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               signed char *ip);

extern int
ncmpi_get_vars_text(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride, char *ip);
extern int
ncmpi_get_vars_text_all(int ncid, int varid, const MPI_Offset *start,
                   const MPI_Offset *count, const MPI_Offset *stride, char *ip);

extern int
ncmpi_get_vars_short(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride, short *ip);
extern int
ncmpi_get_vars_short_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride, short *ip);

extern int
ncmpi_get_vars_int(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride, int *ip);
extern int
ncmpi_get_vars_int_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride, int *ip);

extern int
ncmpi_get_vars_float(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride, float *ip);
extern int
ncmpi_get_vars_float_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride, float *ip);

extern int
ncmpi_get_vars_double(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride, double *ip);
extern int
ncmpi_get_vars_double_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride, double *ip);

extern int
ncmpi_get_vars_longlong(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               long long *ip);
extern int
ncmpi_get_vars_longlong_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               long long *ip);

/* Begin Skip Prototypes for Fortran binding */
/* skip types: uchar, ubyte, ushort, uint, long, ulonglong string */

extern int
ncmpi_put_vars_uchar(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const unsigned char *op);
extern int
ncmpi_put_vars_uchar_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const unsigned char *op);

extern int
ncmpi_put_vars_ushort(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const unsigned short *op);
extern int
ncmpi_put_vars_ushort_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const unsigned short *op);

extern int
ncmpi_put_vars_uint(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const unsigned int *op);
extern int
ncmpi_put_vars_uint_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const unsigned int *op);

extern int
ncmpi_put_vars_long(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const long *op);
extern int
ncmpi_put_vars_long_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const long *op);

extern int
ncmpi_put_vars_ulonglong(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const unsigned long long *op);
extern int
ncmpi_put_vars_ulonglong_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const unsigned long long *op);

extern int
ncmpi_get_vars_uchar(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               unsigned char *ip);
extern int
ncmpi_get_vars_uchar_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               unsigned char *ip);

extern int
ncmpi_get_vars_ushort(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               unsigned short *ip);
extern int
ncmpi_get_vars_ushort_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               unsigned short *ip);

extern int
ncmpi_get_vars_uint(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               unsigned int *ip);
extern int
ncmpi_get_vars_uint_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               unsigned int *ip);

extern int
ncmpi_get_vars_long(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               long *ip);
extern int
ncmpi_get_vars_long_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               long *ip);

extern int
ncmpi_get_vars_ulonglong(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               unsigned long long *ip);
extern int
ncmpi_get_vars_ulonglong_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               unsigned long long *ip);

/* End Skip Prototypes for Fortran binding */

/* End {put,get}_vars */

/* Begin {put,get}_varm */

extern int
ncmpi_put_varm(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, const void *op,
               MPI_Offset bufcount, MPI_Datatype buftype);
extern int
ncmpi_put_varm_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, const void *op,
               MPI_Offset bufcount, MPI_Datatype buftype);

extern int
ncmpi_put_varm_text(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, const char *op);
extern int
ncmpi_put_varm_text_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, const char *op);

extern int
ncmpi_put_varm_schar(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, const signed char *op);
extern int
ncmpi_put_varm_schar_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, const signed char *op);

extern int
ncmpi_put_varm_short(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, const short *op);
extern int
ncmpi_put_varm_short_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, const short *op);

extern int
ncmpi_put_varm_int(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, const int *op);
extern int
ncmpi_put_varm_int_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, const int *op);

extern int
ncmpi_put_varm_float(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, const float *op);
extern int
ncmpi_put_varm_float_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, const float *op);

extern int
ncmpi_put_varm_double(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, const double *op);
extern int
ncmpi_put_varm_double_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, const double *op);

extern int
ncmpi_put_varm_longlong(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, const long long *op);
extern int
ncmpi_put_varm_longlong_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, const long long *op);

extern int
ncmpi_get_varm(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, void *ip, MPI_Offset bufcount,
               MPI_Datatype buftype);
extern int
ncmpi_get_varm_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, void *ip, MPI_Offset bufcount,
               MPI_Datatype buftype);

extern int
ncmpi_get_varm_schar(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, signed char *ip);
extern int
ncmpi_get_varm_schar_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, signed char *ip);

extern int
ncmpi_get_varm_text(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, char *ip);
extern int
ncmpi_get_varm_text_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, char *ip);

extern int
ncmpi_get_varm_short(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, short *ip);
extern int
ncmpi_get_varm_short_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, short *ip);

extern int
ncmpi_get_varm_int(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, int *ip);
extern int
ncmpi_get_varm_int_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, int *ip);

extern int
ncmpi_get_varm_float(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, float *ip);
extern int
ncmpi_get_varm_float_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, float *ip);

extern int
ncmpi_get_varm_double(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, double *ip);
extern int
ncmpi_get_varm_double_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, double *ip);

extern int
ncmpi_get_varm_longlong(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, long long *ip);
extern int
ncmpi_get_varm_longlong_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, long long *ip);

/* Begin Skip Prototypes for Fortran binding */
/* skip types: uchar, ubyte, ushort, uint, long, ulonglong string */

extern int
ncmpi_put_varm_uchar(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, const unsigned char *op);
extern int
ncmpi_put_varm_uchar_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, const unsigned char *op);

extern int
ncmpi_put_varm_ushort(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, const unsigned short *op);
extern int
ncmpi_put_varm_ushort_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, const unsigned short *op);

extern int
ncmpi_put_varm_uint(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, const unsigned int *op);
extern int
ncmpi_put_varm_uint_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, const unsigned int *op);

extern int
ncmpi_put_varm_long(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, const long *op);
extern int
ncmpi_put_varm_long_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, const long *op);

extern int
ncmpi_put_varm_ulonglong(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, const unsigned long long *op);
extern int
ncmpi_put_varm_ulonglong_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, const unsigned long long *op);

extern int
ncmpi_get_varm_uchar(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, unsigned char *ip);
extern int
ncmpi_get_varm_uchar_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, unsigned char *ip);

extern int
ncmpi_get_varm_ushort(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, unsigned short *ip);
extern int
ncmpi_get_varm_ushort_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, unsigned short *ip);

extern int
ncmpi_get_varm_uint(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, unsigned int *ip);
extern int
ncmpi_get_varm_uint_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, unsigned int *ip);

extern int
ncmpi_get_varm_long(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, long *ip);
extern int
ncmpi_get_varm_long_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, long *ip);

extern int
ncmpi_get_varm_ulonglong(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, unsigned long long *ip);
extern int
ncmpi_get_varm_ulonglong_all(int ncid, int varid, const MPI_Offset *start,
               const MPI_Offset *count, const MPI_Offset *stride,
               const MPI_Offset *imap, unsigned long long *ip);

/* End Skip Prototypes for Fortran binding */

/* End {put,get}_varm */

/* Begin of {put,get}_varn{kind} */

extern int
ncmpi_put_varn(int ncid, int varid, int num, MPI_Offset* const *starts,
               MPI_Offset* const *counts, const void *op,
               MPI_Offset bufcount, MPI_Datatype buftype);
extern int
ncmpi_put_varn_all(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
                const void *op, MPI_Offset bufcount, MPI_Datatype buftype);

extern int
ncmpi_get_varn(int ncid, int varid, int num, MPI_Offset* const *starts,
               MPI_Offset* const *counts,  void *ip, MPI_Offset bufcount,
               MPI_Datatype buftype);
extern int
ncmpi_get_varn_all(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               void *ip, MPI_Offset bufcount, MPI_Datatype buftype);

extern int
ncmpi_put_varn_text(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const char *op);
extern int
ncmpi_put_varn_text_all(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const char *op);

extern int
ncmpi_put_varn_schar(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const signed char *op);
extern int
ncmpi_put_varn_schar_all(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const signed char *op);

extern int
ncmpi_put_varn_short(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const short *op);
extern int
ncmpi_put_varn_short_all(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const short *op);

extern int
ncmpi_put_varn_int(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const int *op);
extern int
ncmpi_put_varn_int_all(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const int *op);

extern int
ncmpi_put_varn_float(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const float *op);
extern int
ncmpi_put_varn_float_all(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const float *op);

extern int
ncmpi_put_varn_double(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const double *op);
extern int
ncmpi_put_varn_double_all(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const double *op);

extern int
ncmpi_put_varn_longlong(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const long long *op);
extern int
ncmpi_put_varn_longlong_all(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const long long *op);

/* Begin Skip Prototypes for Fortran binding */
/* skip types: uchar, ubyte, ushort, uint, long, ulonglong string */

extern int
ncmpi_put_varn_uchar(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const unsigned char *op);
extern int
ncmpi_put_varn_uchar_all(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const unsigned char *op);

extern int
ncmpi_put_varn_ushort(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const unsigned short *op);
extern int
ncmpi_put_varn_ushort_all(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const unsigned short *op);

extern int
ncmpi_put_varn_uint(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const unsigned int *op);
extern int
ncmpi_put_varn_uint_all(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const unsigned int *op);

extern int
ncmpi_put_varn_long(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const long *op);
extern int
ncmpi_put_varn_long_all(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const long *op);

extern int
ncmpi_put_varn_ulonglong(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const unsigned long long *op);
extern int
ncmpi_put_varn_ulonglong_all(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const unsigned long long *op);

/* End Skip Prototypes for Fortran binding */

extern int
ncmpi_get_varn_text(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               char *ip);
extern int
ncmpi_get_varn_text_all(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               char *ip);

extern int
ncmpi_get_varn_schar(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               signed char *ip);
extern int
ncmpi_get_varn_schar_all(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               signed char *ip);

extern int
ncmpi_get_varn_short(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               short *ip);
extern int
ncmpi_get_varn_short_all(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               short *ip);

extern int
ncmpi_get_varn_int(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               int *ip);
extern int
ncmpi_get_varn_int_all(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               int *ip);

extern int
ncmpi_get_varn_float(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               float *ip);
extern int
ncmpi_get_varn_float_all(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               float *ip);

extern int
ncmpi_get_varn_double(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               double *ip);
extern int
ncmpi_get_varn_double_all(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               double *ip);

extern int
ncmpi_get_varn_longlong(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               long long *ip);
extern int
ncmpi_get_varn_longlong_all(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               long long *ip);

/* Begin Skip Prototypes for Fortran binding */
/* skip types: uchar, ubyte, ushort, uint, long, ulonglong string */

extern int
ncmpi_get_varn_uchar(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               unsigned char *ip);
extern int
ncmpi_get_varn_uchar_all(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               unsigned char *ip);

extern int
ncmpi_get_varn_ushort(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               unsigned short *ip);
extern int
ncmpi_get_varn_ushort_all(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               unsigned short *ip);

extern int
ncmpi_get_varn_uint(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               unsigned int *ip);
extern int
ncmpi_get_varn_uint_all(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               unsigned int *ip);

extern int
ncmpi_get_varn_long(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               long *ip);
extern int
ncmpi_get_varn_long_all(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               long *ip);

extern int
ncmpi_get_varn_ulonglong(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               unsigned long long *ip);
extern int
ncmpi_get_varn_ulonglong_all(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               unsigned long long *ip);

/* End Skip Prototypes for Fortran binding */

/* End of {put,get}_varn{kind} */

/* Begin {put,get}_vard */
extern int
ncmpi_get_vard(int ncid, int varid, MPI_Datatype filetype, void *ip,
               MPI_Offset bufcount, MPI_Datatype buftype);
extern int
ncmpi_get_vard_all(int ncid, int varid, MPI_Datatype filetype, void *ip,
               MPI_Offset bufcount, MPI_Datatype buftype);
extern int
ncmpi_put_vard(int ncid, int varid, MPI_Datatype filetype, const void *ip,
               MPI_Offset bufcount, MPI_Datatype buftype);
extern int
ncmpi_put_vard_all(int ncid, int varid, MPI_Datatype filetype, const void *ip,
               MPI_Offset bufcount, MPI_Datatype buftype);
/* End of {put,get}_vard */

/* Begin {mput,mget}_var */

/* #################################################################### */
/* Begin: more prototypes to be included for Fortran binding conversion */

/* Begin non-blocking data access functions */

extern int
ncmpi_wait(int ncid, int count, int array_of_requests[],
           int array_of_statuses[]);

extern int
ncmpi_wait_all(int ncid, int count, int array_of_requests[],
               int array_of_statuses[]);

extern int
ncmpi_cancel(int ncid, int num, int *reqs, int *statuses);

extern int
ncmpi_buffer_attach(int ncid, MPI_Offset bufsize);
extern int
ncmpi_buffer_detach(int ncid);
extern int
ncmpi_inq_buffer_usage(int ncid, MPI_Offset *usage);
extern int
ncmpi_inq_buffer_size(int ncid, MPI_Offset *buf_size);
extern int
ncmpi_inq_nreqs(int ncid, int *nreqs);

/* Begin {iput,iget,bput}_var1 */

extern int
ncmpi_iput_var1(int ncid, int varid, const MPI_Offset *start,
                const void *op, MPI_Offset bufcount,
                MPI_Datatype buftype, int *req);

extern int
ncmpi_iput_var1_text(int ncid, int varid, const MPI_Offset *start,
                const char *op, int *req);

extern int
ncmpi_iput_var1_schar(int ncid, int varid, const MPI_Offset *start,
                const signed char *op, int *req);

extern int
ncmpi_iput_var1_short(int ncid, int varid, const MPI_Offset *start,
                const short *op, int *req);

extern int
ncmpi_iput_var1_int(int ncid, int varid, const MPI_Offset *start,
                const int *op, int *req);

extern int
ncmpi_iput_var1_float(int ncid, int varid, const MPI_Offset *start,
                const float *op, int *req);

extern int
ncmpi_iput_var1_double(int ncid, int varid, const MPI_Offset *start,
                const double *op, int *req);

extern int
ncmpi_iput_var1_longlong(int ncid, int varid, const MPI_Offset *start,
                const long long *op, int *req);

extern int
ncmpi_iget_var1(int ncid, int varid, const MPI_Offset *start, void *ip,
                MPI_Offset bufcount, MPI_Datatype buftype, int *req);

extern int
ncmpi_iget_var1_schar(int ncid, int varid, const MPI_Offset *start,
                signed char *ip, int *req);

extern int
ncmpi_iget_var1_text(int ncid, int varid, const MPI_Offset *start,
                char *ip, int *req);

extern int
ncmpi_iget_var1_short(int ncid, int varid, const MPI_Offset *start,
                short *ip, int *req);

extern int
ncmpi_iget_var1_int(int ncid, int varid, const MPI_Offset *start,
                int *ip, int *req);

extern int
ncmpi_iget_var1_float(int ncid, int varid, const MPI_Offset *start,
                float *ip, int *req);

extern int
ncmpi_iget_var1_double(int ncid, int varid, const MPI_Offset *start,
                double *ip, int *req);

extern int
ncmpi_iget_var1_longlong(int ncid, int varid, const MPI_Offset *start,
                long long *ip, int *req);

extern int
ncmpi_bput_var1(int ncid, int varid, const MPI_Offset *start, const void *op,
                MPI_Offset bufcount, MPI_Datatype buftype, int *req);

extern int
ncmpi_bput_var1_text(int ncid, int varid, const MPI_Offset *start,
                const char *op, int *req);

extern int
ncmpi_bput_var1_schar(int ncid, int varid, const MPI_Offset *start,
                const signed char *op, int *req);

extern int
ncmpi_bput_var1_short(int ncid, int varid, const MPI_Offset *start,
                const short *op, int *req);

extern int
ncmpi_bput_var1_int(int ncid, int varid, const MPI_Offset *start,
                const int *op, int *req);

extern int
ncmpi_bput_var1_float(int ncid, int varid, const MPI_Offset *start,
                const float *op, int *req);

extern int
ncmpi_bput_var1_double(int ncid, int varid, const MPI_Offset *start,
                const double *op, int *req);

extern int
ncmpi_bput_var1_longlong(int ncid, int varid, const MPI_Offset *start,
                const long long *op, int *req);

/* Begin Skip Prototypes for Fortran binding */
/* skip types: uchar, ubyte, ushort, uint, long, ulonglong string */

extern int
ncmpi_iput_var1_uchar(int ncid, int varid, const MPI_Offset *start,
                const unsigned char *op, int *req);

extern int
ncmpi_iput_var1_ushort(int ncid, int varid, const MPI_Offset *start,
                const unsigned short *op, int *req);

extern int
ncmpi_iput_var1_uint(int ncid, int varid, const MPI_Offset *start,
                const unsigned int *op, int *req);

extern int
ncmpi_iput_var1_long(int ncid, int varid, const MPI_Offset *start,
                const long *ip, int *req);

extern int
ncmpi_iput_var1_ulonglong(int ncid, int varid, const MPI_Offset *start,
                const unsigned long long *op, int *req);

extern int
ncmpi_iget_var1_uchar(int ncid, int varid, const MPI_Offset *start,
                unsigned char *ip, int *req);

extern int
ncmpi_iget_var1_ushort(int ncid, int varid, const MPI_Offset *start,
                unsigned short *ip, int *req);

extern int
ncmpi_iget_var1_uint(int ncid, int varid, const MPI_Offset *start,
                unsigned int *ip, int *req);

extern int
ncmpi_iget_var1_long(int ncid, int varid, const MPI_Offset *start,
                long *ip, int *req);

extern int
ncmpi_iget_var1_ulonglong(int ncid, int varid, const MPI_Offset *start,
                unsigned long long *ip, int *req);

extern int
ncmpi_bput_var1_uchar(int ncid, int varid, const MPI_Offset *start,
                const unsigned char *op, int *req);

extern int
ncmpi_bput_var1_ushort(int ncid, int varid, const MPI_Offset *start,
                const unsigned short *op, int *req);

extern int
ncmpi_bput_var1_uint(int ncid, int varid, const MPI_Offset *start,
                const unsigned int *op, int *req);

extern int
ncmpi_bput_var1_long(int ncid, int varid, const MPI_Offset *start,
                const long *ip, int *req);

extern int
ncmpi_bput_var1_ulonglong(int ncid, int varid, const MPI_Offset *start,
                const unsigned long long *op, int *req);

/* End Skip Prototypes for Fortran binding */

/* End {iput,iget,bput}_var1 */

/* Begin {iput,iget,bput}_var */

extern int
ncmpi_iput_var(int ncid, int varid, const void *op, MPI_Offset bufcount,
               MPI_Datatype buftype, int *req);

extern int
ncmpi_iput_var_schar(int ncid, int varid, const signed char *op, int *req);

extern int
ncmpi_iput_var_text(int ncid, int varid, const char *op, int *req);

extern int
ncmpi_iput_var_short(int ncid, int varid, const short *op, int *req);

extern int
ncmpi_iput_var_int(int ncid, int varid, const int *op, int *req);

extern int
ncmpi_iput_var_float(int ncid, int varid, const float *op, int *req);

extern int
ncmpi_iput_var_double(int ncid, int varid, const double *op, int *req);

extern int
ncmpi_iput_var_longlong(int ncid, int varid, const long long *op, int *req);

extern int
ncmpi_iget_var(int ncid, int varid, void *ip, MPI_Offset bufcount,
               MPI_Datatype buftype, int *req);

extern int
ncmpi_iget_var_schar(int ncid, int varid, signed char *ip, int *req);

extern int
ncmpi_iget_var_text(int ncid, int varid, char *ip, int *req);

extern int
ncmpi_iget_var_short(int ncid, int varid, short *ip, int *req);

extern int
ncmpi_iget_var_int(int ncid, int varid, int *ip, int *req);

extern int
ncmpi_iget_var_float(int ncid, int varid, float *ip, int *req);

extern int
ncmpi_iget_var_double(int ncid, int varid, double *ip, int *req);

extern int
ncmpi_iget_var_longlong(int ncid, int varid, long long *ip, int *req);

extern int
ncmpi_bput_var(int ncid, int varid, const void *op, MPI_Offset bufcount,
               MPI_Datatype buftype, int *req);

extern int
ncmpi_bput_var_schar(int ncid, int varid, const signed char *op, int *req);

extern int
ncmpi_bput_var_text(int ncid, int varid, const char *op, int *req);

extern int
ncmpi_bput_var_short(int ncid, int varid, const short *op, int *req);

extern int
ncmpi_bput_var_int(int ncid, int varid, const int *op, int *req);

extern int
ncmpi_bput_var_float(int ncid, int varid, const float *op, int *req);

extern int
ncmpi_bput_var_double(int ncid, int varid, const double *op, int *req);

extern int
ncmpi_bput_var_longlong(int ncid, int varid, const long long *op, int *req);

/* Begin Skip Prototypes for Fortran binding */
/* skip types: uchar, ubyte, ushort, uint, long, ulonglong string */

extern int
ncmpi_iput_var_uchar(int ncid, int varid, const unsigned char *op, int *req);

extern int
ncmpi_iput_var_ushort(int ncid, int varid, const unsigned short *op, int *req);

extern int
ncmpi_iput_var_uint(int ncid, int varid, const unsigned int *op, int *req);

extern int
ncmpi_iput_var_long(int ncid, int varid, const long *op, int *req);

extern int
ncmpi_iput_var_ulonglong(int ncid, int varid, const unsigned long long *op,
               int *req);

extern int
ncmpi_iget_var_uchar(int ncid, int varid, unsigned char *ip, int *req);

extern int
ncmpi_iget_var_ushort(int ncid, int varid, unsigned short *ip, int *req);

extern int
ncmpi_iget_var_uint(int ncid, int varid, unsigned int *ip, int *req);

extern int
ncmpi_iget_var_long(int ncid, int varid, long *ip, int *req);

extern int
ncmpi_iget_var_ulonglong(int ncid, int varid, unsigned long long *ip, int *req);

extern int
ncmpi_bput_var_uchar(int ncid, int varid, const unsigned char *op, int *req);

extern int
ncmpi_bput_var_ushort(int ncid, int varid, const unsigned short *op, int *req);

extern int
ncmpi_bput_var_uint(int ncid, int varid, const unsigned int *op, int *req);

extern int
ncmpi_bput_var_long(int ncid, int varid, const long *op, int *req);

extern int
ncmpi_bput_var_ulonglong(int ncid, int varid, const unsigned long long *op,
               int *req);

/* End Skip Prototypes for Fortran binding */

/* End {iput,iget,bput}_var */

/* Begin {iput,iget,bput}_vara */

extern int
ncmpi_iput_vara(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const void *op,
                MPI_Offset bufcount, MPI_Datatype buftype, int *req);

extern int
ncmpi_iput_vara_schar(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const signed char *op, int *req);

extern int
ncmpi_iput_vara_text(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const char *op, int *req);

extern int
ncmpi_iput_vara_short(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const short *op, int *req);

extern int
ncmpi_iput_vara_int(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const int *op, int *req);

extern int
ncmpi_iput_vara_float(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const float *op, int *req);

extern int
ncmpi_iput_vara_double(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const double *op, int *req);

extern int
ncmpi_iput_vara_longlong(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const long long *op, int *req);

extern int
ncmpi_iget_vara(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, void *ip, MPI_Offset bufcount,
                MPI_Datatype buftype, int *req);

extern int
ncmpi_iget_vara_schar(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, signed char *ip, int *req);

extern int
ncmpi_iget_vara_text(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, char *ip, int *req);

extern int
ncmpi_iget_vara_short(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, short *ip, int *req);

extern int
ncmpi_iget_vara_int(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, int *ip, int *req);

extern int
ncmpi_iget_vara_float(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, float *ip, int *req);

extern int
ncmpi_iget_vara_double(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, double *ip, int *req);

extern int
ncmpi_iget_vara_longlong(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, long long *ip, int *req);

extern int
ncmpi_bput_vara(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const void *op,
                MPI_Offset bufcount, MPI_Datatype buftype, int *req);

extern int
ncmpi_bput_vara_schar(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const signed char *op, int *req);

extern int
ncmpi_bput_vara_text(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const char *op, int *req);

extern int
ncmpi_bput_vara_short(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const short *op, int *req);

extern int
ncmpi_bput_vara_int(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const int *op, int *req);

extern int
ncmpi_bput_vara_float(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const float *op, int *req);

extern int
ncmpi_bput_vara_double(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const double *op, int *req);

extern int
ncmpi_bput_vara_longlong(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const long long *op, int *req);

/* Begin Skip Prototypes for Fortran binding */
/* skip types: uchar, ubyte, ushort, uint, long, ulonglong string */

extern int
ncmpi_iput_vara_uchar(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const unsigned char *op, int *req);

extern int
ncmpi_iput_vara_ushort(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const unsigned short *op, int *req);

extern int
ncmpi_iput_vara_uint(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const unsigned int *op, int *req);

extern int
ncmpi_iput_vara_long(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const long *op, int *req);

extern int
ncmpi_iput_vara_ulonglong(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const unsigned long long *op,
                int *req);

extern int
ncmpi_iget_vara_uchar(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, unsigned char *ip, int *req);

extern int
ncmpi_iget_vara_ushort(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, unsigned short *ip, int *req);

extern int
ncmpi_iget_vara_uint(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, unsigned int *ip, int *req);

extern int
ncmpi_iget_vara_long(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, long *ip, int *req);

extern int
ncmpi_iget_vara_ulonglong(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, unsigned long long *ip, int *req);

extern int
ncmpi_bput_vara_uchar(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const unsigned char *op, int *req);

extern int
ncmpi_bput_vara_ushort(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const unsigned short *op, int *req);

extern int
ncmpi_bput_vara_uint(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const unsigned int *op, int *req);

extern int
ncmpi_bput_vara_long(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const long *op, int *req);

extern int
ncmpi_bput_vara_ulonglong(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const unsigned long long *op,
                int *req);

/* End Skip Prototypes for Fortran binding */

/* End {iput,iget,bput}_vara */

/* Begin {iput,iget,bput}_vars */

extern int
ncmpi_iput_vars(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const void *op, MPI_Offset bufcount,
                MPI_Datatype buftype, int *req);

extern int
ncmpi_iput_vars_schar(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const signed char *op, int *req);

extern int
ncmpi_iput_vars_text(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const char *op, int *req);

extern int
ncmpi_iput_vars_short(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const short *op, int *req);

extern int
ncmpi_iput_vars_int(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const int *op, int *req);

extern int
ncmpi_iput_vars_float(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const float *op, int *req);

extern int
ncmpi_iput_vars_double(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const double *op, int *req);

extern int
ncmpi_iput_vars_longlong(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const long long *op, int *req);

extern int
ncmpi_iget_vars(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride, void *ip,
                MPI_Offset bufcount, MPI_Datatype buftype, int *req);

extern int
ncmpi_iget_vars_schar(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                signed char *ip, int *req);

extern int
ncmpi_iget_vars_text(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                char *ip, int *req);

extern int
ncmpi_iget_vars_short(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                short *ip, int *req);

extern int
ncmpi_iget_vars_int(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                int *ip, int *req);

extern int
ncmpi_iget_vars_float(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                float *ip, int *req);

extern int
ncmpi_iget_vars_double(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                double *ip, int *req);

extern int
ncmpi_iget_vars_longlong(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                long long *ip, int *req);

extern int
ncmpi_bput_vars(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const void *op, MPI_Offset bufcount,
                MPI_Datatype buftype, int *req);

extern int
ncmpi_bput_vars_schar(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const signed char *op, int *req);

extern int
ncmpi_bput_vars_text(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const char *op, int *req);

extern int
ncmpi_bput_vars_short(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const short *op, int *req);

extern int
ncmpi_bput_vars_int(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const int *op, int *req);

extern int
ncmpi_bput_vars_float(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const float *op, int *req);

extern int
ncmpi_bput_vars_double(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const double *op, int *req);

extern int
ncmpi_bput_vars_longlong(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const long long *op, int *req);

/* Begin Skip Prototypes for Fortran binding */
/* skip types: uchar, ubyte, ushort, uint, long, ulonglong string */

extern int
ncmpi_iput_vars_uchar(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const unsigned char *op, int *req);

extern int
ncmpi_iput_vars_ushort(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const unsigned short *op, int *req);

extern int
ncmpi_iput_vars_uint(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const unsigned int *op, int *req);

extern int
ncmpi_iput_vars_long(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const long *op, int *req);

extern int
ncmpi_iput_vars_ulonglong(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const unsigned long long *op, int *req);

extern int
ncmpi_iget_vars_uchar(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                unsigned char *ip, int *req);

extern int
ncmpi_iget_vars_ushort(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                unsigned short *ip, int *req);

extern int
ncmpi_iget_vars_uint(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                unsigned int *ip, int *req);

extern int
ncmpi_iget_vars_long(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                long *ip, int *req);

extern int
ncmpi_iget_vars_ulonglong(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                unsigned long long *ip, int *req);

extern int
ncmpi_bput_vars_uchar(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const unsigned char *op, int *req);

extern int
ncmpi_bput_vars_ushort(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const unsigned short *op, int *req);

extern int
ncmpi_bput_vars_uint(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const unsigned int *op, int *req);

extern int
ncmpi_bput_vars_long(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const long *op, int *req);

extern int
ncmpi_bput_vars_ulonglong(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const unsigned long long *op, int *req);

/* End Skip Prototypes for Fortran binding */

/* End {iput,iget,bput}_vars */

/* Begin {iput,iget,bput}_varm */

extern int
ncmpi_iput_varm(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, const void *op,
                MPI_Offset bufcount, MPI_Datatype buftype, int *req);

extern int
ncmpi_iput_varm_schar(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, const signed char *op,
                int *req);

extern int
ncmpi_iput_varm_text(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, const char *op, int *req);

extern int
ncmpi_iput_varm_short(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, const short *op, int *req);

extern int
ncmpi_iput_varm_int(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, const int *op, int *req);

extern int
ncmpi_iput_varm_float(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, const float *op, int *req);

extern int
ncmpi_iput_varm_double(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, const double *op, int *req);

extern int
ncmpi_iput_varm_longlong(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, const long long *op, int *req);

extern int
ncmpi_iget_varm(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, void *ip, MPI_Offset bufcount,
                MPI_Datatype buftype, int *req);

extern int
ncmpi_iget_varm_schar(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, signed char *ip, int *req);

extern int
ncmpi_iget_varm_text(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, char *ip, int *req);

extern int
ncmpi_iget_varm_short(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, short *ip, int *req);

extern int
ncmpi_iget_varm_int(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, int *ip, int *req);

extern int
ncmpi_iget_varm_float(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, float *ip, int *req);

extern int
ncmpi_iget_varm_double(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, double *ip, int *req);

extern int
ncmpi_iget_varm_longlong(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, long long *ip, int *req);

extern int
ncmpi_bput_varm(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, const void *op,
                MPI_Offset bufcount, MPI_Datatype buftype, int *req);

extern int
ncmpi_bput_varm_schar(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, const signed char *op, int *req);

extern int
ncmpi_bput_varm_text(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, const char *op, int *req);

extern int
ncmpi_bput_varm_short(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, const short *op, int *req);

extern int
ncmpi_bput_varm_int(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, const int *op, int *req);

extern int
ncmpi_bput_varm_float(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, const float *op, int *req);

extern int
ncmpi_bput_varm_double(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, const double *op, int *req);

extern int
ncmpi_bput_varm_longlong(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, const long long *op, int *req);

/* Begin Skip Prototypes for Fortran binding */
/* skip types: uchar, ubyte, ushort, uint, long, ulonglong string */

extern int
ncmpi_iput_varm_uchar(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, const unsigned char *op, int *req);

extern int
ncmpi_iput_varm_ushort(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, const unsigned short *op, int *req);

extern int
ncmpi_iput_varm_uint(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, const unsigned int *op, int *req);

extern int
ncmpi_iput_varm_long(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, const long *op, int *req);

extern int
ncmpi_iput_varm_ulonglong(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, const unsigned long long *op,
                int *req);

extern int
ncmpi_iget_varm_uchar(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, unsigned char *ip, int *req);

extern int
ncmpi_iget_varm_ushort(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, unsigned short *ip, int *req);

extern int
ncmpi_iget_varm_uint(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, unsigned int *ip, int *req);

extern int
ncmpi_iget_varm_long(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, long *ip, int *req);

extern int
ncmpi_iget_varm_ulonglong(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, unsigned long long *ip, int *req);

extern int
ncmpi_bput_varm_uchar(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, const unsigned char *op,
                int *req);

extern int
ncmpi_bput_varm_ushort(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, const unsigned short *op,
                int *req);

extern int
ncmpi_bput_varm_uint(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, const unsigned int *op,
                int *req);

extern int
ncmpi_bput_varm_long(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, const long *op, int *req);

extern int
ncmpi_bput_varm_ulonglong(int ncid, int varid, const MPI_Offset *start,
                const MPI_Offset *count, const MPI_Offset *stride,
                const MPI_Offset *imap, const unsigned long long *op,
                int *req);

/* End Skip Prototypes for Fortran binding */

/* End {iput,iget,bput}_varm */

/* Begin of nonblocking {iput,iget}_varn{kind} */

extern int
ncmpi_iput_varn(int ncid, int varid, int num, MPI_Offset* const *starts,
                MPI_Offset* const *counts, const void *op,
                MPI_Offset bufcount, MPI_Datatype buftype, int *req);

extern int
ncmpi_iget_varn(int ncid, int varid, int num, MPI_Offset* const *starts,
                MPI_Offset* const *counts,  void *op, MPI_Offset bufcount,
                MPI_Datatype buftype, int *req);

extern int
ncmpi_iput_varn_text(int ncid, int varid, int num,
                MPI_Offset* const *starts,
                MPI_Offset* const *counts, const char *op, int *req);

extern int
ncmpi_iput_varn_schar(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const signed char *op, int *req);

extern int
ncmpi_iput_varn_short(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const short *op, int *req);

extern int
ncmpi_iput_varn_int(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const int *op, int *req);

extern int
ncmpi_iput_varn_float(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const float *op, int *req);

extern int
ncmpi_iput_varn_double(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const double *op, int *req);

extern int
ncmpi_iput_varn_longlong(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const long long *op, int *req);


/* Begin Skip Prototypes for Fortran binding */
/* skip types: uchar, ubyte, ushort, uint, long, ulonglong string */

extern int
ncmpi_iput_varn_uchar(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const unsigned char *op, int *req);

extern int
ncmpi_iput_varn_ushort(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const unsigned short *op, int *req);

extern int
ncmpi_iput_varn_uint(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const unsigned int *op, int *req);

extern int
ncmpi_iput_varn_long(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const long *op, int *req);

extern int
ncmpi_iput_varn_ulonglong(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const unsigned long long *op, int *req);

/* End Skip Prototypes for Fortran binding */

extern int
ncmpi_iget_varn_text(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               char *ip, int *req);

extern int
ncmpi_iget_varn_schar(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               signed char *ip, int *req);

extern int
ncmpi_iget_varn_short(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               short *ip, int *req);

extern int
ncmpi_iget_varn_int(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               int *ip, int *req);

extern int
ncmpi_iget_varn_float(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               float *ip, int *req);

extern int
ncmpi_iget_varn_double(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               double *ip, int *req);

extern int
ncmpi_iget_varn_longlong(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               long long *ip, int *req);


/* Begin Skip Prototypes for Fortran binding */
/* skip types: uchar, ubyte, ushort, uint, long, ulonglong string */

extern int
ncmpi_iget_varn_uchar(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               unsigned char *ip, int *req);

extern int
ncmpi_iget_varn_ushort(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               unsigned short *ip, int *req);

extern int
ncmpi_iget_varn_uint(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               unsigned int *ip, int *req);

extern int
ncmpi_iget_varn_long(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               long *ip, int *req);

extern int
ncmpi_iget_varn_ulonglong(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               unsigned long long *ip, int *req);

/* End Skip Prototypes for Fortran binding */

/* End of {iput,iget}_varn{kind} */

/* Begin of nonblocking bput_varn{kind} */

extern int
ncmpi_bput_varn(int ncid, int varid, int num, MPI_Offset* const *starts,
               MPI_Offset* const *counts, const void *op,
               MPI_Offset bufcount, MPI_Datatype buftype, int *req);

extern int
ncmpi_bput_varn_text(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const char *op, int *req);

extern int
ncmpi_bput_varn_schar(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const signed char *op, int *req);

extern int
ncmpi_bput_varn_short(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const short *op, int *req);

extern int
ncmpi_bput_varn_int(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const int *op, int *req);

extern int
ncmpi_bput_varn_float(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const float *op, int *req);

extern int
ncmpi_bput_varn_double(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const double *op, int *req);

extern int
ncmpi_bput_varn_longlong(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const long long *op, int *req);


/* Begin Skip Prototypes for Fortran binding */
/* skip types: uchar, ubyte, ushort, uint, long, ulonglong string */

extern int
ncmpi_bput_varn_uchar(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const unsigned char *op, int *req);

extern int
ncmpi_bput_varn_ushort(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const unsigned short *op, int *req);

extern int
ncmpi_bput_varn_uint(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const unsigned int *op, int *req);

extern int
ncmpi_bput_varn_long(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const long *op, int *req);

extern int
ncmpi_bput_varn_ulonglong(int ncid, int varid, int num,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               const unsigned long long *op, int *req);

/* End Skip Prototypes for Fortran binding */

/* End of bput_varn{kind} */

/* End non-blocking data access functions */

/* Begin Skip Prototypes for Fortran binding */
/* skip all mput/mget APIs as Fortran cannot handle array of buffers */

extern int
ncmpi_mput_var(int ncid, int num, int *varids, void* const *buf,
               const MPI_Offset *bufcounts, const MPI_Datatype datatypes[]);

extern int
ncmpi_mput_var_all(int ncid, int num, int *varids, void* const *buf,
               const MPI_Offset *bufcounts, const MPI_Datatype datatypes[]);

extern int
ncmpi_mput_var_text(int ncid, int num, int *varids, char* const *buf);
extern int
ncmpi_mput_var_text_all(int ncid, int num, int *varids, char* const *buf);

extern int
ncmpi_mput_var_schar(int ncid, int num, int *varids, signed char* const *buf);
extern int
ncmpi_mput_var_schar_all(int ncid, int num, int *varids, signed char* const *buf);

extern int
ncmpi_mput_var_uchar(int ncid, int num, int *varids, unsigned char* const *buf);
extern int
ncmpi_mput_var_uchar_all(int ncid, int num, int *varids, unsigned char* const *buf);

extern int
ncmpi_mput_var_short(int ncid, int num, int *varids, short* const *buf);
extern int
ncmpi_mput_var_short_all(int ncid, int num, int *varids, short* const *buf);

extern int
ncmpi_mput_var_ushort(int ncid, int num, int *varids, unsigned short* const *buf);
extern int
ncmpi_mput_var_ushort_all(int ncid, int num, int *varids,
               unsigned short* const *buf);

extern int
ncmpi_mput_var_int(int ncid, int num, int *varids, int* const *buf);
extern int
ncmpi_mput_var_int_all(int ncid, int num, int *varids, int* const *buf);

extern int
ncmpi_mput_var_uint(int ncid, int num, int *varids, unsigned int* const *buf);
extern int
ncmpi_mput_var_uint_all(int ncid, int num, int *varids, unsigned int* const *buf);

extern int
ncmpi_mput_var_long(int ncid, int num, int *varids, long* const *buf);
extern int
ncmpi_mput_var_long_all(int ncid, int num, int *varids, long* const *buf);

extern int
ncmpi_mput_var_float(int ncid, int num, int *varids, float* const *buf);
extern int
ncmpi_mput_var_float_all(int ncid, int num, int *varids, float* const *buf);

extern int
ncmpi_mput_var_double(int ncid, int num, int *varids, double* const *buf);
extern int
ncmpi_mput_var_double_all(int ncid, int num, int *varids, double* const *buf);

extern int
ncmpi_mput_var_longlong(int ncid, int num, int *varids, long long* const *buf);
extern int
ncmpi_mput_var_longlong_all(int ncid, int num, int *varids, long long* const *buf);

extern int
ncmpi_mput_var_ulonglong(int ncid, int num, int *varids,
               unsigned long long* const *buf);
extern int
ncmpi_mput_var_ulonglong_all(int ncid, int num, int *varids,
               unsigned long long* const *buf);

extern int
ncmpi_mput_var1(int ncid, int num, int *varids,
               MPI_Offset* const *starts, void* const *buf,
               const MPI_Offset *bufcounts, const MPI_Datatype datatypes[]);
extern int
ncmpi_mput_var1_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, void* const *buf,
               const MPI_Offset *bufcounts, const MPI_Datatype datatypes[]);

extern int
ncmpi_mput_var1_text(int ncid, int num, int *varids,
               MPI_Offset* const *starts, char* const *buf);
extern int
ncmpi_mput_var1_text_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, char* const *buf);

extern int
ncmpi_mput_var1_schar(int ncid, int num, int *varids,
               MPI_Offset* const *starts, signed char* const *buf);
extern int
ncmpi_mput_var1_schar_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, signed char* const *buf);

extern int
ncmpi_mput_var1_uchar(int ncid, int num, int *varids,
               MPI_Offset* const *starts, unsigned char* const *buf);
extern int
ncmpi_mput_var1_uchar_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, unsigned char* const *buf);

extern int
ncmpi_mput_var1_short(int ncid, int num, int *varids,
               MPI_Offset* const *starts, short* const *buf);
extern int
ncmpi_mput_var1_short_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, short* const *buf);

extern int
ncmpi_mput_var1_ushort(int ncid, int num, int *varids,
               MPI_Offset* const *starts, unsigned short* const *buf);
extern int
ncmpi_mput_var1_ushort_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, unsigned short* const *buf);

extern int
ncmpi_mput_var1_int(int ncid, int num, int *varids,
               MPI_Offset* const *starts, int* const *buf);
extern int
ncmpi_mput_var1_int_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, int* const *buf);

extern int
ncmpi_mput_var1_uint(int ncid, int num, int *varids,
               MPI_Offset* const *starts, unsigned int* const *buf);
extern int
ncmpi_mput_var1_uint_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, unsigned int* const *buf);

extern int
ncmpi_mput_var1_long(int ncid, int num, int *varids,
               MPI_Offset* const *starts, long* const *buf);
extern int
ncmpi_mput_var1_long_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, long* const *buf);

extern int
ncmpi_mput_var1_float(int ncid, int num, int *varids,
               MPI_Offset* const *starts, float* const *buf);
extern int
ncmpi_mput_var1_float_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, float* const *buf);

extern int
ncmpi_mput_var1_double(int ncid, int num, int *varids,
               MPI_Offset* const *starts, double* const *buf);
extern int
ncmpi_mput_var1_double_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, double* const *buf);

extern int
ncmpi_mput_var1_longlong(int ncid, int num, int *varids,
               MPI_Offset* const *starts, long long* const *buf);
extern int
ncmpi_mput_var1_longlong_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, long long* const *buf);

extern int
ncmpi_mput_var1_ulonglong(int ncid, int num, int *varids,
               MPI_Offset* const *starts, unsigned long long* const *buf);
extern int
ncmpi_mput_var1_ulonglong_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, unsigned long long* const *buf);


extern int
ncmpi_mput_vara(int ncid, int num, int *varids, MPI_Offset* const *starts,
               MPI_Offset* const *counts, void* const *buf,
               const MPI_Offset *bufcounts, const MPI_Datatype datatypes[]);
extern int
ncmpi_mput_vara_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               void* const *buf, const MPI_Offset *bufcounts, const MPI_Datatype datatypes[]);

extern int
ncmpi_mput_vara_text(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               char* const *buf);
extern int
ncmpi_mput_vara_text_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               char* const *buf);

extern int
ncmpi_mput_vara_schar(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               signed char* const *buf);
extern int
ncmpi_mput_vara_schar_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               signed char* const *buf);

extern int
ncmpi_mput_vara_uchar(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               unsigned char* const *buf);
extern int
ncmpi_mput_vara_uchar_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               unsigned char* const *buf);

extern int
ncmpi_mput_vara_short(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               short* const *buf);
extern int
ncmpi_mput_vara_short_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               short* const *buf);

extern int
ncmpi_mput_vara_ushort(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               unsigned short* const *buf);
extern int
ncmpi_mput_vara_ushort_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               unsigned short* const *buf);

extern int
ncmpi_mput_vara_int(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               int* const *buf);
extern int
ncmpi_mput_vara_int_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               int* const *buf);

extern int
ncmpi_mput_vara_uint(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               unsigned int* const *buf);
extern int
ncmpi_mput_vara_uint_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               unsigned int* const *buf);

extern int
ncmpi_mput_vara_long(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               long* const *buf);
extern int
ncmpi_mput_vara_long_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               long* const *buf);

extern int
ncmpi_mput_vara_float(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               float* const *buf);
extern int
ncmpi_mput_vara_float_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               float* const *buf);

extern int
ncmpi_mput_vara_double(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               double* const *buf);
extern int
ncmpi_mput_vara_double_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               double* const *buf);

extern int
ncmpi_mput_vara_longlong(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               long long* const *buf);
extern int
ncmpi_mput_vara_longlong_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               long long* const *buf);

extern int
ncmpi_mput_vara_ulonglong(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               unsigned long long* const *buf);
extern int
ncmpi_mput_vara_ulonglong_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               unsigned long long* const *buf);


extern int
ncmpi_mput_vars(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, void* const *buf,
               const MPI_Offset *bufcounts, const MPI_Datatype datatypes[]);

extern int
ncmpi_mput_vars_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, void* const *buf,
               const MPI_Offset *bufcounts, const MPI_Datatype datatypes[]);

extern int
ncmpi_mput_vars_text(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, char* const *buf);
extern int
ncmpi_mput_vars_text_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, char* const *buf);

extern int
ncmpi_mput_vars_schar(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, signed char* const *buf);
extern int
ncmpi_mput_vars_schar_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, signed char* const *buf);

extern int
ncmpi_mput_vars_uchar(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, unsigned char* const *buf);
extern int
ncmpi_mput_vars_uchar_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, unsigned char* const *buf);

extern int
ncmpi_mput_vars_short(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, short* const *buf);
extern int
ncmpi_mput_vars_short_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, short* const *buf);

extern int
ncmpi_mput_vars_ushort(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, unsigned short* const *buf);
extern int
ncmpi_mput_vars_ushort_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, unsigned short* const *buf);

extern int
ncmpi_mput_vars_int(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, int* const *buf);
extern int
ncmpi_mput_vars_int_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, int* const *buf);

extern int
ncmpi_mput_vars_uint(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, unsigned int* const *buf);
extern int
ncmpi_mput_vars_uint_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, unsigned int* const *buf);

extern int
ncmpi_mput_vars_long(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, long* const *buf);
extern int
ncmpi_mput_vars_long_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, long* const *buf);

extern int
ncmpi_mput_vars_float(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, float* const *buf);
extern int
ncmpi_mput_vars_float_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, float* const *buf);

extern int
ncmpi_mput_vars_double(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, double* const *buf);
extern int
ncmpi_mput_vars_double_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, double* const *buf);

extern int
ncmpi_mput_vars_longlong(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, long long* const *buf);
extern int
ncmpi_mput_vars_longlong_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, long long* const *buf);

extern int
ncmpi_mput_vars_ulonglong(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, unsigned long long* const *buf);
extern int
ncmpi_mput_vars_ulonglong_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, unsigned long long* const *buf);

extern int
ncmpi_mput_varm(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               void* const *buf, const MPI_Offset *bufcounts, const MPI_Datatype datatypes[]);
extern int
ncmpi_mput_varm_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               void* const *buf, const MPI_Offset *bufcounts, const MPI_Datatype datatypes[]);

extern int
ncmpi_mput_varm_text(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               char* const *buf);
extern int
ncmpi_mput_varm_text_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               char* const *buf);

extern int
ncmpi_mput_varm_schar(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               signed char* const *buf);
extern int
ncmpi_mput_varm_schar_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               signed char* const *buf);

extern int
ncmpi_mput_varm_uchar(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               unsigned char* const *buf);
extern int
ncmpi_mput_varm_uchar_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               unsigned char* const *buf);

extern int
ncmpi_mput_varm_short(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               short* const *buf);
extern int
ncmpi_mput_varm_short_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               short* const *buf);

extern int
ncmpi_mput_varm_ushort(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               unsigned short* const *buf);
extern int
ncmpi_mput_varm_ushort_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               unsigned short* const *buf);

extern int
ncmpi_mput_varm_int(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               int* const *buf);
extern int
ncmpi_mput_varm_int_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               int* const *buf);

extern int
ncmpi_mput_varm_uint(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               unsigned int* const *buf);
extern int
ncmpi_mput_varm_uint_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               unsigned int* const *buf);

extern int
ncmpi_mput_varm_long(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               long* const *buf);
extern int
ncmpi_mput_varm_long_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               long* const *buf);

extern int
ncmpi_mput_varm_float(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               float* const *buf);
extern int
ncmpi_mput_varm_float_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               float* const *buf);

extern int
ncmpi_mput_varm_double(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               double* const *buf);
extern int
ncmpi_mput_varm_double_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               double* const *buf);

extern int
ncmpi_mput_varm_longlong(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               long long* const *buf);
extern int
ncmpi_mput_varm_longlong_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               long long* const *buf);

extern int
ncmpi_mput_varm_ulonglong(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               unsigned long long* const *buf);
extern int
ncmpi_mput_varm_ulonglong_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               unsigned long long* const *buf);

extern int
ncmpi_mget_var(int ncid, int num, int *varids, void *bufs[],
               const MPI_Offset *bufcounts, const MPI_Datatype *datatypes);

extern int
ncmpi_mget_var_all(int ncid, int num, int *varids, void *bufs[],
               const MPI_Offset *bufcounts, const MPI_Datatype *datatypes);

extern int
ncmpi_mget_var_text(int ncid, int num, int *varids, char *bufs[]);
extern int
ncmpi_mget_var_text_all(int ncid, int num, int *varids, char *bufs[]);

extern int
ncmpi_mget_var_schar(int ncid, int num, int *varids, signed char *bufs[]);
extern int
ncmpi_mget_var_schar_all(int ncid, int num, int *varids, signed char *bufs[]);

extern int
ncmpi_mget_var_uchar(int ncid, int num, int *varids, unsigned char *bufs[]);
extern int
ncmpi_mget_var_uchar_all(int ncid, int num, int *varids, unsigned char *bufs[]);

extern int
ncmpi_mget_var_short(int ncid, int num, int *varids, short *bufs[]);
extern int
ncmpi_mget_var_short_all(int ncid, int num, int *varids, short *bufs[]);

extern int
ncmpi_mget_var_ushort(int ncid, int num, int *varids, unsigned short *bufs[]);
extern int
ncmpi_mget_var_ushort_all(int ncid, int num, int *varids,
               unsigned short *bufs[]);

extern int
ncmpi_mget_var_int(int ncid, int num, int *varids, int *bufs[]);
extern int
ncmpi_mget_var_int_all(int ncid, int num, int *varids, int *bufs[]);

extern int
ncmpi_mget_var_uint(int ncid, int num, int *varids, unsigned int *bufs[]);
extern int
ncmpi_mget_var_uint_all(int ncid, int num, int *varids, unsigned int *bufs[]);

extern int
ncmpi_mget_var_long(int ncid, int num, int *varids, long *bufs[]);
extern int
ncmpi_mget_var_long_all(int ncid, int num, int *varids, long *bufs[]);

extern int
ncmpi_mget_var_float(int ncid, int num, int *varids, float *bufs[]);
extern int
ncmpi_mget_var_float_all(int ncid, int num, int *varids, float *bufs[]);

extern int
ncmpi_mget_var_double(int ncid, int num, int *varids, double *bufs[]);
extern int
ncmpi_mget_var_double_all(int ncid, int num, int *varids, double *bufs[]);

extern int
ncmpi_mget_var_longlong(int ncid, int num, int *varids, long long *bufs[]);
extern int
ncmpi_mget_var_longlong_all(int ncid, int num, int *varids, long long *bufs[]);

extern int
ncmpi_mget_var_ulonglong(int ncid, int num, int *varids,
               unsigned long long *bufs[]);
extern int
ncmpi_mget_var_ulonglong_all(int ncid, int num, int *varids,
               unsigned long long *bufs[]);

extern int
ncmpi_mget_var1(int ncid, int num, int *varids,
               MPI_Offset* const *starts, void *bufs[],
               const MPI_Offset *bufcounts, const MPI_Datatype *datatypes);
extern int
ncmpi_mget_var1_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, void *bufs[],
               const MPI_Offset *bufcounts, const MPI_Datatype *datatypes);

extern int
ncmpi_mget_var1_text(int ncid, int num, int *varids,
               MPI_Offset* const *starts, char *bufs[]);
extern int
ncmpi_mget_var1_text_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, char *bufs[]);

extern int
ncmpi_mget_var1_schar(int ncid, int num, int *varids,
               MPI_Offset* const *starts, signed char *bufs[]);
extern int
ncmpi_mget_var1_schar_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, signed char *bufs[]);

extern int
ncmpi_mget_var1_uchar(int ncid, int num, int *varids,
               MPI_Offset* const *starts, unsigned char *bufs[]);
extern int
ncmpi_mget_var1_uchar_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, unsigned char *bufs[]);

extern int
ncmpi_mget_var1_short(int ncid, int num, int *varids,
               MPI_Offset* const *starts, short *bufs[]);
extern int
ncmpi_mget_var1_short_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, short *bufs[]);

extern int
ncmpi_mget_var1_ushort(int ncid, int num, int *varids,
               MPI_Offset* const *starts, unsigned short *bufs[]);
extern int
ncmpi_mget_var1_ushort_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, unsigned short *bufs[]);

extern int
ncmpi_mget_var1_int(int ncid, int num, int *varids,
               MPI_Offset* const *starts, int *bufs[]);
extern int
ncmpi_mget_var1_int_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, int *bufs[]);

extern int
ncmpi_mget_var1_uint(int ncid, int num, int *varids,
               MPI_Offset* const *starts, unsigned int *bufs[]);
extern int
ncmpi_mget_var1_uint_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, unsigned int *bufs[]);

extern int
ncmpi_mget_var1_long(int ncid, int num, int *varids,
               MPI_Offset* const *starts, long *bufs[]);
extern int
ncmpi_mget_var1_long_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, long *bufs[]);

extern int
ncmpi_mget_var1_float(int ncid, int num, int *varids,
               MPI_Offset* const *starts, float *bufs[]);
extern int
ncmpi_mget_var1_float_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, float *bufs[]);

extern int
ncmpi_mget_var1_double(int ncid, int num, int *varids,
               MPI_Offset* const *starts, double *bufs[]);
extern int
ncmpi_mget_var1_double_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, double *bufs[]);

extern int
ncmpi_mget_var1_longlong(int ncid, int num, int *varids,
               MPI_Offset* const *starts, long long *bufs[]);
extern int
ncmpi_mget_var1_longlong_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, long long *bufs[]);

extern int
ncmpi_mget_var1_ulonglong(int ncid, int num, int *varids,
               MPI_Offset* const *starts, unsigned long long *bufs[]);
extern int
ncmpi_mget_var1_ulonglong_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, unsigned long long *bufs[]);


extern int
ncmpi_mget_vara(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               void *bufs[], const MPI_Offset *bufcounts, const MPI_Datatype *datatypes);
extern int
ncmpi_mget_vara_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               void *bufs[], const MPI_Offset *bufcounts, const MPI_Datatype *datatypes);

extern int
ncmpi_mget_vara_text(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               char *bufs[]);
extern int
ncmpi_mget_vara_text_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               char *bufs[]);

extern int
ncmpi_mget_vara_schar(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               signed char *bufs[]);
extern int
ncmpi_mget_vara_schar_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               signed char *bufs[]);

extern int
ncmpi_mget_vara_uchar(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               unsigned char *bufs[]);
extern int
ncmpi_mget_vara_uchar_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               unsigned char *bufs[]);

extern int
ncmpi_mget_vara_short(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               short *bufs[]);
extern int
ncmpi_mget_vara_short_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               short *bufs[]);

extern int
ncmpi_mget_vara_ushort(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               unsigned short *bufs[]);
extern int
ncmpi_mget_vara_ushort_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               unsigned short *bufs[]);

extern int
ncmpi_mget_vara_int(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               int *bufs[]);
extern int
ncmpi_mget_vara_int_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               int *bufs[]);

extern int
ncmpi_mget_vara_uint(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               unsigned int *bufs[]);
extern int
ncmpi_mget_vara_uint_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               unsigned int *bufs[]);

extern int
ncmpi_mget_vara_long(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               long *bufs[]);
extern int
ncmpi_mget_vara_long_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               long *bufs[]);

extern int
ncmpi_mget_vara_float(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               float *bufs[]);
extern int
ncmpi_mget_vara_float_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               float *bufs[]);

extern int
ncmpi_mget_vara_double(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               double *bufs[]);
extern int
ncmpi_mget_vara_double_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               double *bufs[]);

extern int
ncmpi_mget_vara_longlong(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               long long *bufs[]);
extern int
ncmpi_mget_vara_longlong_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               long long *bufs[]);

extern int
ncmpi_mget_vara_ulonglong(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               unsigned long long *bufs[]);
extern int
ncmpi_mget_vara_ulonglong_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               unsigned long long *bufs[]);

extern int
ncmpi_mget_vars(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, void *bufs[],
               const MPI_Offset *bufcounts, const MPI_Datatype *datatypes);
extern int
ncmpi_mget_vars_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, void *bufs[],
               const MPI_Offset *bufcounts, const MPI_Datatype *datatypes);

extern int
ncmpi_mget_vars_text(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, char *bufs[]);
extern int
ncmpi_mget_vars_text_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, char *bufs[]);

extern int
ncmpi_mget_vars_schar(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, signed char *bufs[]);
extern int
ncmpi_mget_vars_schar_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, signed char *bufs[]);

extern int
ncmpi_mget_vars_uchar(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, unsigned char *bufs[]);
extern int
ncmpi_mget_vars_uchar_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, unsigned char *bufs[]);

extern int
ncmpi_mget_vars_short(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, short *bufs[]);
extern int
ncmpi_mget_vars_short_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, short *bufs[]);

extern int
ncmpi_mget_vars_ushort(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, unsigned short *bufs[]);
extern int
ncmpi_mget_vars_ushort_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, unsigned short *bufs[]);

extern int
ncmpi_mget_vars_int(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, int *bufs[]);
extern int
ncmpi_mget_vars_int_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, int *bufs[]);

extern int
ncmpi_mget_vars_uint(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, unsigned int *bufs[]);
extern int
ncmpi_mget_vars_uint_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, unsigned int *bufs[]);

extern int
ncmpi_mget_vars_long(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, long *bufs[]);
extern int
ncmpi_mget_vars_long_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, long *bufs[]);

extern int
ncmpi_mget_vars_float(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, float *bufs[]);
extern int
ncmpi_mget_vars_float_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, float *bufs[]);

extern int
ncmpi_mget_vars_double(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, double *bufs[]);
extern int
ncmpi_mget_vars_double_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, double *bufs[]);

extern int
ncmpi_mget_vars_longlong(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, long long *bufs[]);
extern int
ncmpi_mget_vars_longlong_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, long long *bufs[]);

extern int
ncmpi_mget_vars_ulonglong(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, unsigned long long *bufs[]);
extern int
ncmpi_mget_vars_ulonglong_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, unsigned long long *bufs[]);

extern int
ncmpi_mget_varm(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               void *bufs[], const MPI_Offset *bufcounts, const MPI_Datatype *datatypes);

extern int
ncmpi_mget_varm_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               void *bufs[], const MPI_Offset *bufcounts, const MPI_Datatype *datatypes);

extern int
ncmpi_mget_varm_text(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               char *bufs[]);
extern int
ncmpi_mget_varm_text_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               char *bufs[]);

extern int
ncmpi_mget_varm_schar(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               signed char *bufs[]);
extern int
ncmpi_mget_varm_schar_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               signed char *bufs[]);

extern int
ncmpi_mget_varm_uchar(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               unsigned char *bufs[]);
extern int
ncmpi_mget_varm_uchar_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               unsigned char *bufs[]);

extern int
ncmpi_mget_varm_short(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               short *bufs[]);
extern int
ncmpi_mget_varm_short_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               short *bufs[]);

extern int
ncmpi_mget_varm_ushort(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               unsigned short *bufs[]);
extern int
ncmpi_mget_varm_ushort_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               unsigned short *bufs[]);

extern int
ncmpi_mget_varm_int(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               int *bufs[]);
extern int
ncmpi_mget_varm_int_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               int *bufs[]);

extern int
ncmpi_mget_varm_uint(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               unsigned int *bufs[]);
extern int
ncmpi_mget_varm_uint_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               unsigned int *bufs[]);

extern int
ncmpi_mget_varm_long(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               long *bufs[]);
extern int
ncmpi_mget_varm_long_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               long *bufs[]);

extern int
ncmpi_mget_varm_float(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               float *bufs[]);
extern int
ncmpi_mget_varm_float_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               float *bufs[]);

extern int
ncmpi_mget_varm_double(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               double *bufs[]);
extern int
ncmpi_mget_varm_double_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               double *bufs[]);

extern int
ncmpi_mget_varm_longlong(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               long long *bufs[]);
extern int
ncmpi_mget_varm_longlong_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               long long *bufs[]);

extern int
ncmpi_mget_varm_ulonglong(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               unsigned long long *bufs[]);
extern int
ncmpi_mget_varm_ulonglong_all(int ncid, int num, int *varids,
               MPI_Offset* const *starts, MPI_Offset* const *counts,
               MPI_Offset* const *strides, MPI_Offset* const *imaps,
               unsigned long long *bufs[]);

/* End Skip Prototypes for Fortran binding */

/* End {mput,mget}_var */

/* End: more prototypes to be included for Fortran binding conversion */
/* ################################################################## */

/* End Prototypes */


/* Macros below are defined in serial netcdf (3.5.0) for backwards
 * compatibility with older netcdf code. We aren't concerned with backwards
 * compatibility, so if your code doesn't compile with PnetCDF, maybe
 * this is why:
 *
 *
 *  OLD NAME                 NEW NAME
 *  ----------------------------------
 *  FILL_BYTE       NC_FILL_BYTE
 *  FILL_CHAR       NC_FILL_CHAR
 *  FILL_SHORT      NC_FILL_SHORT
 *  FILL_LONG       NC_FILL_INT
 *  FILL_FLOAT      NC_FILL_FLOAT
 *  FILL_DOUBLE     NC_FILL_DOUBLE
 *
 *  MAX_NC_DIMS     NC_MAX_DIMS
 *  MAX_NC_ATTRS    NC_MAX_ATTRS
 *  MAX_NC_VARS     NC_MAX_VARS
 *  MAX_NC_NAME     NC_MAX_NAME
 *  MAX_VAR_DIMS    NC_MAX_VAR_DIMS
 */

#if defined(__cplusplus)
}
#endif
#endif
