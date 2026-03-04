package qupath.ext.dlclassifier.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.ConcurrentSkipListSet;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

/**
 * Manages a bounded cache of probability maps for tile boundary blending.
 * <p>
 * The cache stores raw probability maps from inference results keyed by
 * tile request coordinates. When blending is requested, it looks up
 * neighboring tiles (left, right, top, bottom) and applies a linear
 * cross-fade at boundaries to eliminate visible seams in the overlay.
 * <p>
 * Also tracks observed tile positions to compute the empirical step
 * between tiles, which is used to locate neighbor tiles in the cache.
 * <p>
 * After the initial batch of tiles is cached, a debounced one-shot
 * overlay refresh is scheduled so that all tiles get re-rendered with
 * proper bidirectional blending.
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class TileBlendCache {

    private static final Logger logger = LoggerFactory.getLogger(TileBlendCache.class);

    /** Cache of probability maps. Key = (requestX, requestY) packed into long. */
    private final ConcurrentHashMap<Long, float[][][]> probCache = new ConcurrentHashMap<>();

    /** Tracks insertion order for LRU eviction. */
    private final ConcurrentLinkedDeque<Long> probCacheOrder = new ConcurrentLinkedDeque<>();

    /** Maximum cached probability maps. */
    private final int maxSize;

    /** Observed tile request X positions for empirical step computation. */
    private final ConcurrentSkipListSet<Integer> seenTileX = new ConcurrentSkipListSet<>();
    /** Observed tile request Y positions for empirical step computation. */
    private final ConcurrentSkipListSet<Integer> seenTileY = new ConcurrentSkipListSet<>();
    /** Empirical step between tiles in full-res X coords. -1 = unknown. */
    private volatile int empiricalStepX = -1;
    /** Empirical step between tiles in full-res Y coords. -1 = unknown. */
    private volatile int empiricalStepY = -1;

    /** Debounced scheduler for viewer refresh after new tiles are cached. */
    private final ScheduledExecutorService refreshScheduler =
            Executors.newSingleThreadScheduledExecutor(r -> {
                Thread t = new Thread(r, "dl-overlay-refresh");
                t.setDaemon(true);
                return t;
            });
    private volatile ScheduledFuture<?> pendingRefresh;

    /** Guards against repeated overlay refreshes -- only one refresh per session. */
    private volatile boolean hasRefreshed = false;

    /** Callback invoked when a deferred overlay refresh fires. */
    private final Runnable refreshCallback;

    /**
     * Creates a new tile blend cache.
     *
     * @param maxSize         maximum number of probability maps to cache
     * @param refreshCallback called when a deferred overlay refresh fires
     */
    public TileBlendCache(int maxSize, Runnable refreshCallback) {
        this.maxSize = maxSize;
        this.refreshCallback = refreshCallback;
    }

    /**
     * Packs (x, y) request coordinates into a single long key.
     */
    public static long cacheKey(int requestX, int requestY) {
        return ((long) requestX << 32) | (requestY & 0xFFFFFFFFL);
    }

    /**
     * Returns the cached probability map for the given tile coordinates, or null.
     */
    public float[][][] getIfCached(int requestX, int requestY) {
        return probCache.get(cacheKey(requestX, requestY));
    }

    /**
     * Caches a probability map and tracks tile positions for step computation.
     * Evicts the oldest entry if over capacity.
     */
    public void cache(int requestX, int requestY, float[][][] probMap) {
        long key = cacheKey(requestX, requestY);
        probCache.put(key, probMap);
        probCacheOrder.addLast(key);

        // LRU eviction
        while (probCache.size() > maxSize) {
            Long oldest = probCacheOrder.pollFirst();
            if (oldest != null) {
                probCache.remove(oldest);
            }
        }

        // Track tile positions for empirical step computation
        seenTileX.add(requestX);
        seenTileY.add(requestY);
        if (empiricalStepX < 0 || empiricalStepY < 0) {
            computeEmpiricalStep();
        }
    }

    /**
     * Returns the current number of cached probability maps.
     */
    public int size() {
        return probCache.size();
    }

    /**
     * Returns the empirical tile step in X, or -1 if not yet computed.
     */
    public int getEmpiricalStepX() {
        return empiricalStepX;
    }

    /**
     * Returns the empirical tile step in Y, or -1 if not yet computed.
     */
    public int getEmpiricalStepY() {
        return empiricalStepY;
    }

    /**
     * Blends this tile's probability map with cached neighbor probability maps
     * at tile boundaries using a linear cross-fade.
     * <p>
     * For non-overlapping tiles (the common case with QuPath's tiling), adjacent
     * tiles abut without shared pixels. Blending creates a smooth transition by
     * mixing each tile's edge pixels with the neighbor's corresponding edge pixels:
     * <ul>
     *   <li>At the seam (d=0): 50% self + 50% neighbor -- both tiles get identical
     *       probability vectors, so argmax produces the same class on both sides</li>
     *   <li>Deep inside (d=blendDist): 100% self -- unaffected by blending</li>
     * </ul>
     * <p>
     * Horizontal blending (left/right) is applied first, then vertical (top/bottom).
     * This sequential approach handles corners naturally.
     *
     * @param probMap   this tile's raw probability map [height][width][numClasses]
     * @param requestX  tile request X coordinate (full-resolution image coords)
     * @param requestY  tile request Y coordinate (full-resolution image coords)
     * @param width     tile width in pixels
     * @param height    tile height in pixels
     * @return blended probability map (new array, original not modified)
     */
    public float[][][] blendWithNeighbors(float[][][] probMap, int requestX, int requestY,
                                           int width, int height) {
        // Need empirical step to locate neighbors
        int stepX = empiricalStepX;
        int stepY = empiricalStepY;
        if (stepX <= 0 && stepY <= 0) {
            return probMap;  // Step not yet computed, skip blending
        }

        int numClasses = probMap[0][0].length;

        // Look up cached neighbors using empirical step
        float[][][] left   = (stepX > 0) ? probCache.get(cacheKey(requestX - stepX, requestY)) : null;
        float[][][] right  = (stepX > 0) ? probCache.get(cacheKey(requestX + stepX, requestY)) : null;
        float[][][] top    = (stepY > 0) ? probCache.get(cacheKey(requestX, requestY - stepY)) : null;
        float[][][] bottom = (stepY > 0) ? probCache.get(cacheKey(requestX, requestY + stepY)) : null;

        if (left == null && right == null && top == null && bottom == null) {
            return probMap;  // No neighbors available
        }

        // Blend distance: pixels on each side of boundary participating in cross-fade.
        // ~6% of tile width per side (12% total). Enough to smooth seams without
        // distorting predictions deep inside the tile.
        int blendDist = Math.max(8, Math.min(32, Math.min(width, height) / 16));

        // Create a copy for blending (don't modify cached original)
        float[][][] blended = deepCopyProbMap(probMap);

        // --- Horizontal blending (uses probMap as self source) ---

        // Right boundary: blend this tile's right edge with right neighbor's left edge
        if (right != null) {
            int rh = Math.min(height, right.length);
            for (int y = 0; y < rh; y++) {
                for (int d = 0; d < blendDist; d++) {
                    int x = width - 1 - d;   // this tile: from right edge inward
                    int nx = d;               // neighbor: from left edge inward
                    if (x < 0 || nx >= right[y].length) break;
                    // d=0 (at edge): wSelf=0.5, d=blendDist-1 (deep inside): wSelf~=1.0
                    float t = (float) d / blendDist;
                    float wSelf = 0.5f + 0.5f * t;
                    float wNeighbor = 1.0f - wSelf;
                    int nc = Math.min(numClasses, right[y][nx].length);
                    for (int c = 0; c < nc; c++) {
                        blended[y][x][c] = wSelf * probMap[y][x][c]
                                + wNeighbor * right[y][nx][c];
                    }
                }
            }
        }

        // Left boundary: blend this tile's left edge with left neighbor's right edge
        if (left != null) {
            int lh = Math.min(height, left.length);
            for (int y = 0; y < lh; y++) {
                int leftW = left[y].length;
                for (int d = 0; d < blendDist; d++) {
                    int x = d;                    // this tile: from left edge inward
                    int nx = leftW - 1 - d;       // neighbor: from right edge inward
                    if (nx < 0) break;
                    float t = (float) d / blendDist;
                    float wSelf = 0.5f + 0.5f * t;
                    float wNeighbor = 1.0f - wSelf;
                    int nc = Math.min(numClasses, left[y][nx].length);
                    for (int c = 0; c < nc; c++) {
                        blended[y][x][c] = wSelf * probMap[y][x][c]
                                + wNeighbor * left[y][nx][c];
                    }
                }
            }
        }

        // --- Vertical blending (uses blended as self source for corner handling) ---

        // Bottom boundary: blend this tile's bottom edge with bottom neighbor's top edge
        if (bottom != null) {
            for (int d = 0; d < blendDist; d++) {
                int y = height - 1 - d;   // this tile: from bottom edge inward
                int ny = d;               // neighbor: from top edge inward
                if (y < 0 || ny >= bottom.length) break;
                float t = (float) d / blendDist;
                float wSelf = 0.5f + 0.5f * t;
                float wNeighbor = 1.0f - wSelf;
                int bw = Math.min(width, bottom[ny].length);
                for (int x = 0; x < bw; x++) {
                    int nc = Math.min(numClasses, bottom[ny][x].length);
                    for (int c = 0; c < nc; c++) {
                        blended[y][x][c] = wSelf * blended[y][x][c]
                                + wNeighbor * bottom[ny][x][c];
                    }
                }
            }
        }

        // Top boundary: blend this tile's top edge with top neighbor's bottom edge
        if (top != null) {
            int topH = top.length;
            for (int d = 0; d < blendDist; d++) {
                int y = d;                    // this tile: from top edge inward
                int ny = topH - 1 - d;       // neighbor: from bottom edge inward
                if (ny < 0) break;
                float t = (float) d / blendDist;
                float wSelf = 0.5f + 0.5f * t;
                float wNeighbor = 1.0f - wSelf;
                int tw = Math.min(width, top[ny].length);
                for (int x = 0; x < tw; x++) {
                    int nc = Math.min(numClasses, top[ny][x].length);
                    for (int c = 0; c < nc; c++) {
                        blended[y][x][c] = wSelf * blended[y][x][c]
                                + wNeighbor * top[ny][x][c];
                    }
                }
            }
        }

        return blended;
    }

    /**
     * Schedules a debounced, one-shot overlay refresh after the initial tile batch.
     * <p>
     * On the first render, tiles are computed without all neighbors cached, so blending
     * is incomplete. After the batch completes (debounced 1s after the last tile), the
     * overlay is recreated to force fresh tile requests. The cache-hit fast path serves
     * these re-requests instantly from the prob cache, now with all neighbors available
     * for proper bidirectional blending.
     * <p>
     * Only fires once per overlay session to avoid infinite refresh loops.
     */
    public void scheduleRefresh() {
        if (hasRefreshed) {
            logger.debug("BLEND scheduleRefresh skipped (already refreshed)");
            return;
        }

        ScheduledFuture<?> prev = pendingRefresh;
        if (prev != null) prev.cancel(false);
        logger.debug("BLEND scheduling refresh in 1s (cache size={})", probCache.size());
        pendingRefresh = refreshScheduler.schedule(() -> {
            hasRefreshed = true;
            try {
                logger.debug("Refreshing overlay for tile blending ({} cached prob maps)",
                        probCache.size());
                refreshCallback.run();
            } catch (Exception e) {
                logger.debug("Deferred overlay refresh failed: {}", e.getMessage());
            }
        }, 1000, TimeUnit.MILLISECONDS);
    }

    /**
     * Clears all cached data and resets state.
     */
    public void clear() {
        probCache.clear();
        probCacheOrder.clear();
        seenTileX.clear();
        seenTileY.clear();
        empiricalStepX = -1;
        empiricalStepY = -1;
        hasRefreshed = false;
    }

    /**
     * Clears all data and shuts down the refresh scheduler.
     */
    public void shutdown() {
        clear();
        ScheduledFuture<?> pending = pendingRefresh;
        if (pending != null) pending.cancel(false);
        refreshScheduler.shutdownNow();
    }

    // ==================== Internal Methods ====================

    /**
     * Computes the empirical step between tile requests by finding the minimum
     * non-zero gap between observed tile positions.
     */
    private void computeEmpiricalStep() {
        if (seenTileX.size() >= 2 && empiricalStepX < 0) {
            int minGap = Integer.MAX_VALUE;
            Integer prev = null;
            for (Integer pos : seenTileX) {
                if (prev != null) {
                    int gap = pos - prev;
                    if (gap > 0 && gap < minGap) minGap = gap;
                }
                prev = pos;
            }
            if (minGap != Integer.MAX_VALUE) {
                empiricalStepX = minGap;
                logger.info("BLEND empirical stepX = {} (from {} positions)",
                        minGap, seenTileX.size());
            }
        }
        if (seenTileY.size() >= 2 && empiricalStepY < 0) {
            int minGap = Integer.MAX_VALUE;
            Integer prev = null;
            for (Integer pos : seenTileY) {
                if (prev != null) {
                    int gap = pos - prev;
                    if (gap > 0 && gap < minGap) minGap = gap;
                }
                prev = pos;
            }
            if (minGap != Integer.MAX_VALUE) {
                empiricalStepY = minGap;
                logger.info("BLEND empirical stepY = {} (from {} positions)",
                        minGap, seenTileY.size());
            }
        }
    }

    /**
     * Creates a deep copy of a probability map so blending doesn't modify cached data.
     */
    private static float[][][] deepCopyProbMap(float[][][] src) {
        int h = src.length;
        float[][][] copy = new float[h][][];
        for (int y = 0; y < h; y++) {
            int w = src[y].length;
            copy[y] = new float[w][];
            for (int x = 0; x < w; x++) {
                copy[y][x] = src[y][x].clone();
            }
        }
        return copy;
    }
}
