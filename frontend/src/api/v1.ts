/**
 * The version prefix, in one place.
 *
 * `API_BASE` is `/api` and **health sits outside the version prefix** at `/api/health`
 * (ADR-013, and `backend/src/api/app.py` mounts `health_router` with `prefix="/api"` while
 * the versioned router carries its own `/api/v1`). That asymmetry is real and has already
 * cost this project time - an assumed `/health` returned a 404 whose error body satisfied
 * the assertion that was looking for a refusal. So the prefix is a constant rather than a
 * string repeated in eleven client modules, and `routes.test.ts` is not where it is checked:
 * each client's own test asserts the path it calls.
 */
export const V1 = '/v1';
