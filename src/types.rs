// --------- SmallSlice<T> ------------------------------------------------------------
pub(crate) const SMALL_SIZE: usize = 8;
pub(crate) type SmallSlice<T> = smallvec::SmallVec<[T; SMALL_SIZE]>;
